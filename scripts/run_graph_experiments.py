"""Run Phases 10-16 on the available, case-level RE2-TT graph data."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.graph_models import GraphAnomalyRCA


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | None]:
    pred = (scores >= threshold).astype("int8")
    return {
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None,
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def load_graphs(root: Path, split: pd.DataFrame) -> list[Data]:
    graph_root = root / "data/processed/graphs/re2tt/60s"
    embed_root = root / "data/processed/embeddings/qwen/60s"
    case_meta = split.set_index("case")["split"].to_dict()
    roots = pd.read_parquet(root / "data/raw/cases.parquet").set_index("case")["root_cause_service"].to_dict()
    graphs: list[Data] = []
    for case in sorted(case_meta):
        node_path, edge_path = graph_root / case / "nodes.parquet", graph_root / case / "edges.parquet"
        if not node_path.exists() or not edge_path.exists():
            continue
        nodes = pd.read_parquet(node_path)
        edges = pd.read_parquet(edge_path)
        emb_index_path, emb_array_path = embed_root / f"{case}.parquet", embed_root / f"{case}.npy"
        emb_index = pd.read_parquet(emb_index_path) if emb_index_path.exists() else pd.DataFrame()
        emb_array = np.load(emb_array_path, mmap_mode="r") if emb_array_path.exists() else None
        embedding_dim = int(emb_array.shape[1]) if emb_array is not None else 3584
        for window_id, frame in nodes.groupby("window_id", sort=True):
            frame = frame.sort_values("node_id")
            lookup = {str(service): int(pos) for pos, service in enumerate(frame.service)}
            x = np.zeros((len(frame), embedding_dim + 3), dtype=np.float32)
            if emb_array is not None and not emb_index.empty:
                selected = emb_index[emb_index.window_id == window_id]
                for row_pos, row in selected.iterrows():
                    node_pos = lookup.get(str(row.service))
                    if node_pos is not None:
                        x[node_pos, :embedding_dim] = np.asarray(emb_array[row_pos], dtype=np.float32)
            counts = frame[["log_count", "error_count", "warning_count"]].to_numpy(dtype=np.float32)
            denom = np.maximum(counts[:, [0]], 1.0)
            x[:, embedding_dim:] = np.concatenate([np.log1p(counts[:, :1]), counts[:, 1:] / denom], axis=1)
            edge_frame = edges[edges.window_id == window_id]
            edge_index = torch.tensor(edge_frame[["src", "dst"]].to_numpy(dtype=np.int64).T, dtype=torch.long)
            target = torch.tensor([float(str(service) == str(roots.get(case))) for service in frame.service], dtype=torch.float32)
            graphs.append(Data(
                x=torch.from_numpy(x), edge_index=edge_index,
                y=torch.tensor([float(frame.anomaly_label.iloc[0])], dtype=torch.float32),
                rca_target=target, case=case, window_id=int(window_id),
                split=case_meta[case], services=list(frame.service.astype(str)),
                root_cause=str(roots.get(case)), fault=str(split.loc[split.case == case, "fault"].iloc[0]),
            ))
    if not graphs:
        raise RuntimeError("no graph snapshots available")
    return graphs


def run_model(kind: str, graphs: list[Data], config: dict, device: torch.device) -> tuple[dict, pd.DataFrame, list[dict], list[np.ndarray]]:
    settings = config["graph_experiments"]
    train_graphs = [g for g in graphs if g.split == "train"]
    valid_graphs = [g for g in graphs if g.split == "validation"]
    test_graphs = [g for g in graphs if g.split == "test"]
    input_size = int(graphs[0].x.shape[1])
    model = GraphAnomalyRCA(input_size, int(config["model"]["hidden_size"]), int(config["model"]["heads"]), int(config["model"]["layers"]), float(config["model"]["dropout"]), kind).to(device)
    labels = torch.tensor([float(g.y.item()) for g in train_graphs], device=device)
    positive = labels.sum().clamp(min=1.0)
    negative = (len(labels) - labels.sum()).clamp(min=1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=negative / positive)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    loader = DataLoader(train_graphs, batch_size=int(settings["batch_size"]), shuffle=True)
    started = time.perf_counter()
    history = []
    for epoch in range(int(settings["epochs"])):
        model.train(); total = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            anomaly, rca, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(anomaly, batch.y.view(-1)) + float(settings["rca_loss_weight"]) * nn.functional.binary_cross_entropy_with_logits(rca, batch.rca_target)
            loss.backward(); optimizer.step(); total += float(loss.item())
        history.append({"epoch": epoch + 1, "loss": total / max(len(loader), 1)})

    def predict(items: list[Data], collect_attention: bool = False) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        model.eval(); scores, rca_scores, attention_scores = [], [], []
        with torch.no_grad():
            for batch in DataLoader(items, batch_size=int(settings["batch_size"]), shuffle=False):
                batch = batch.to(device); anomaly, rca, attention = model(batch.x, batch.edge_index, batch.batch, return_attention=collect_attention)
                scores.extend(torch.sigmoid(anomaly).cpu().numpy());
                if collect_attention:
                    attention_scores.extend([weights.detach().cpu().numpy() for weights in attention])
                for graph_idx in range(int(batch.y.shape[0])):
                    rca_scores.append(torch.sigmoid(rca)[batch.batch == graph_idx].cpu().numpy())
        return np.asarray(scores), rca_scores, attention_scores

    valid_scores, _, _ = predict(valid_graphs); test_scores, test_rca, attention_scores = predict(test_graphs, collect_attention=(kind == "gatv2"))
    test_labels = np.asarray([g.y.item() for g in test_graphs], dtype=np.int8)
    checkpoint_dir = Path(__file__).resolve().parents[1] / "results/graph_experiments"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "kind": kind, "input_size": input_size, "seed": int(config.get("seed", 42)), "config": config["model"]}, checkpoint_dir / f"phase{10 if kind == 'gcn' else 11 if kind == 'gat' else 12}_{kind}.pt")
    pred_rows = pd.DataFrame({"model": kind, "case": [g.case for g in test_graphs], "window_id": [g.window_id for g in test_graphs], "fault": [g.fault for g in test_graphs], "label": test_labels, "score": test_scores})
    by_case: dict[str, dict[str, object]] = defaultdict(dict)
    for graph, scores in zip(test_graphs, test_rca):
        entry = by_case[graph.case]
        entry.setdefault("root_cause", graph.root_cause); entry.setdefault("scores", defaultdict(float))
        for service, score in zip(graph.services, scores): entry["scores"][service] = max(float(entry["scores"][service]), float(score))
    rca_rows = []
    for case, entry in by_case.items():
        ranking = [service for service, _ in sorted(entry["scores"].items(), key=lambda item: item[1], reverse=True)]
        rank = ranking.index(entry["root_cause"]) + 1 if entry["root_cause"] in ranking else None
        rca_rows.append({"model": kind, "case": case, "root_cause": entry["root_cause"], "rank": rank, "top1": int(rank is not None and rank <= 1), "top3": int(rank is not None and rank <= 3), "top5": int(rank is not None and rank <= 5), "mrr": 1.0 / rank if rank else 0.0})
    rca = pd.DataFrame(rca_rows)
    summary = {"model": kind, "fit_seconds": round(time.perf_counter() - started, 3), "validation": metrics(np.asarray([g.y.item() for g in valid_graphs], dtype=np.int8), valid_scores, float(settings["threshold"])), "test": metrics(test_labels, test_scores, float(settings["threshold"])), "rca": {"cases": int(len(rca)), "top1": float(rca.top1.mean()) if len(rca) else None, "top3": float(rca.top3.mean()) if len(rca) else None, "top5": float(rca.top5.mean()) if len(rca) else None, "mrr": float(rca.mrr.mean()) if len(rca) else None}, "history": history}
    return summary, pred_rows, rca.to_dict(orient="records"), attention_scores


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/default.yaml"); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; config = yaml.safe_load((root / args.config).read_text(encoding="utf-8")); seed = int(config.get("seed", 42)); seed_everything(seed)
    split = pd.read_parquet(root / "data/processed/splits/re2tt_case_split.parquet"); graphs = load_graphs(root, split); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = root / "results/graph_experiments"; output.mkdir(parents=True, exist_ok=True)
    summaries, predictions, rca_rows, gatv2_attention = [], [], [], []
    for kind in ("gcn", "gat", "gatv2"):
        summary, pred, rca, attention = run_model(kind, graphs, config, device); summaries.append(summary); predictions.append(pred); rca_rows.extend(rca)
        if kind == "gatv2": gatv2_attention = attention
    (output / "phase10_12_metrics.json").write_text(json.dumps({"dataset": "RE2-TT", "seed": seed, "device": str(device), "graphs": len(graphs), "models": summaries, "config_file": str((root / args.config).relative_to(root))}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(output / "phase10_12_test_predictions.csv", index=False); pd.DataFrame(rca_rows).to_csv(output / "phase10_12_rca_rankings.csv", index=False)
    if gatv2_attention:
        np.savez_compressed(output / "phase12_gatv2_attention.npz", **{f"graph_{i}": value for i, value in enumerate(gatv2_attention)})
    test_predictions = pd.concat(predictions, ignore_index=True)
    fault_rows = []
    for (model_name, fault), frame in test_predictions.groupby(["model", "fault"], sort=True):
        fault_rows.append({"model": model_name, "fault": fault, "windows": int(len(frame)), **metrics(frame.label.to_numpy(dtype=np.int8), frame.score.to_numpy(dtype=np.float64), float(config["graph_experiments"]["threshold"]))})
    pd.DataFrame(fault_rows).to_csv(output / "phase14_fault_metrics.csv", index=False)
    phase_status = {
        "phase13": {"status": "completed", "description": "GCN/GAT/GATv2 anomaly and RCA comparison", "source": "phase10_12_metrics.json"},
        "phase14": {"status": "completed", "description": "fault-type test metrics generated from real predictions", "output": "phase14_fault_metrics.csv"},
        "phase15": {"status": "not_applicable", "description": "cross-system analysis requires RE2-OB or RE2-SS; only RE2-TT is present"},
        "phase16": {"status": "completed", "description": "representative test-case rankings exported", "source": "phase10_12_rca_rankings.csv"},
    }
    (output / "phase13_16_status.json").write_text(json.dumps(phase_status, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = ["# Phases 10-16 Graph Experiments", "", "| Model | Test F1 | Test ROC-AUC | Test PR-AUC | RCA Top-1 | RCA Top-3 | RCA Top-5 | MRR |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        test_metrics, rca_metrics = item["test"], item["rca"]
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(item["model"], test_metrics["f1"], test_metrics["roc_auc"] or 0.0, test_metrics["pr_auc"], rca_metrics["top1"] or 0.0, rca_metrics["top3"] or 0.0, rca_metrics["top5"] or 0.0, rca_metrics["mrr"] or 0.0))
    lines.extend(["", "Phase 15 is not applicable because only RE2-TT is available; no cross-system result was fabricated."])
    (output / "phase10_16_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"graphs": len(graphs), "models": summaries, "phase13_16": phase_status}, indent=2, ensure_ascii=True))


if __name__ == "__main__": main()
