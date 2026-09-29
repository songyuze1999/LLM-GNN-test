"""Run the Phase 9 frozen-Qwen embedding plus MLP anomaly experiment."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def metric_dict(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    predictions = (scores >= 0.5).astype("int8")
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None,
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_samples(root: Path, split: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    missing: list[str] = []
    for case_row in split.sort_values("case").itertuples(index=False):
        stem = root / "data/processed/embeddings/qwen/60s" / str(case_row.case)
        index_path, array_path = stem.with_suffix(".parquet"), stem.with_suffix(".npy")
        if not index_path.exists() or not array_path.exists():
            missing.append(str(case_row.case))
            continue
        index = pd.read_parquet(index_path, columns=["case", "window_id", "anomaly_label", "phase"])
        array = np.load(array_path, mmap_mode="r")
        if len(index) != len(array):
            raise ValueError(f"embedding/index row mismatch for {case_row.case}")
        for window_id, positions in index.groupby("window_id", sort=True).groups.items():
            pos = np.asarray(list(positions), dtype=np.int64)
            rows.append({
                "case": str(case_row.case), "window_id": int(window_id),
                "anomaly_label": int(index.iloc[pos]["anomaly_label"].max()),
                "phase": str(index.iloc[pos]["phase"].iloc[0]), "split": str(case_row.split),
            })
            vectors.append(np.asarray(array[pos].mean(axis=0), dtype=np.float32))
    if not rows:
        raise RuntimeError("no cached Qwen windows available")
    return pd.DataFrame(rows), np.stack(vectors), missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    split_path = root / "data/processed/splits/re2tt_case_split.parquet"
    split = pd.read_parquet(split_path)
    if split.case.duplicated().any() or split.split.isna().any():
        raise ValueError("case split contains duplicate or unassigned cases")
    samples, vectors, missing = load_samples(root, split)
    sample_cases = set(samples.case)
    split_cases = set(split.case)
    if not sample_cases.issubset(split_cases):
        raise ValueError(f"embedding cases absent from split: {sorted(sample_cases - split_cases)}")
    indices = {part: samples.index[samples.split == part].to_numpy() for part in ("train", "validation", "test")}
    if any(len(indices[part]) == 0 for part in indices):
        raise ValueError(f"empty split: { {part: len(value) for part, value in indices.items()} }")
    labels = samples.anomaly_label.to_numpy(dtype="int8")
    baseline = config["baselines"]
    model = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(int(baseline["mlp_hidden_size"]),),
            max_iter=int(baseline["mlp_max_iter"]),
            early_stopping=bool(baseline["mlp_early_stopping"]),
            random_state=seed,
        )),
    ])
    started = time.perf_counter()
    model.fit(vectors[indices["train"]], labels[indices["train"]])
    fit_seconds = round(time.perf_counter() - started, 3)
    scores = {part: model.predict_proba(vectors[indices[part]])[:, 1] for part in indices}
    metrics = {part: metric_dict(labels[indices[part]], scores[part]) for part in indices}
    output = root / "results/qwen_mlp"
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    for part in ("validation", "test"):
        frame = samples.iloc[indices[part]][["case", "window_id", "anomaly_label", "phase"]].copy()
        frame["split"] = part
        frame["score"] = scores[part]
        prediction_rows.append(frame)
    pd.concat(prediction_rows, ignore_index=True).to_csv(output / "phase9_predictions.csv", index=False)
    manifest_path = root / "data/processed/embeddings/qwen/60s/manifest.json"
    payload = {
        "phase": 9, "dataset": config["data"]["dataset"], "seed": seed, "window_seconds": 60,
        "model": "frozen Qwen mean-pooled service-window embeddings + standardized MLP",
        "embedding_manifest": str(manifest_path.relative_to(root)),
        "embedding_dimension": int(vectors.shape[1]), "cases_used": int(samples.case.nunique()),
        "windows": {part: int(len(indices[part])) for part in indices},
        "missing_embedding_cases": sorted(missing), "fit_seconds": fit_seconds,
        "metrics": metrics, "config_file": str((root / args.config).relative_to(root)),
        "split_file": str(split_path.relative_to(root)),
        "gpu": {"available": bool(torch.cuda.is_available()), "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    (output / "phase9_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output / "phase9_run_metadata.json").write_text(json.dumps({"seed": seed, "split": payload["windows"], "missing_embedding_cases": sorted(missing), "embedding_manifest": payload["embedding_manifest"]}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    table = [
        "# Phase 9: Qwen + MLP",
        "",
        "| Split | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for part in ("train", "validation", "test"):
        values = metrics[part]
        table.append("| {} | {:.6f} | {:.6f} | {:.6f} | {} | {:.6f} |".format(
            part, values["precision"], values["recall"], values["f1"],
            "NA" if values["roc_auc"] is None else f'{values["roc_auc"]:.6f}', values["pr_auc"],
        ))
    (output / "phase9_metrics.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
