"""Build reproducible service graph snapshots from windowed RCAEval traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TRACE_COLUMNS = {
    "window_id", "window_start", "window_end", "caller", "callee",
    "call_count", "error_count", "case", "relative_start", "relative_end",
    "phase", "anomaly_label", "root_cause_service", "fault",
}
LOG_COLUMNS = {
    "case", "window_id", "window_start", "window_end", "relative_start",
    "relative_end", "phase", "anomaly_label", "service", "log_count",
    "error_count", "warning_count", "root_cause_service", "fault",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--window-seconds", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_case_split(index: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Keep the three repeated injections of each root-cause/fault group together."""
    cases = index.loc[index["dataset"] == "RE2-TT", [
        "case", "system", "root_cause_service", "fault", "repetition",
    ]].copy()
    cases["split_group"] = (
        cases["system"].astype(str) + "|" +
        cases["root_cause_service"].astype(str) + "|" +
        cases["fault"].astype(str)
    )
    groups = sorted(cases["split_group"].unique())
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)
    assignments = {
        group: ("train" if pos < n_train else "validation" if pos < n_train + n_valid else "test")
        for pos, group in enumerate(groups)
    }
    cases["split"] = cases["split_group"].map(assignments)
    return cases.sort_values("case").reset_index(drop=True)


def build_case(case: str, source_root: Path, output_root: Path) -> dict[str, object]:
    source = source_root / case / "60s"
    trace_path = source / "trace_edges.parquet"
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    traces = pd.read_parquet(trace_path)
    missing = TRACE_COLUMNS.difference(traces.columns)
    if missing:
        raise ValueError(f"{trace_path} missing columns: {sorted(missing)}")
    traces = traces.copy()
    traces["caller"] = traces["caller"].astype(str)
    traces["callee"] = traces["callee"].astype(str)
    if (traces["caller"] == traces["callee"]).any():
        raise ValueError(f"self-loop found after trace preprocessing: {trace_path}")

    log_path = source / "log_windows.parquet"
    if log_path.exists():
        logs = pd.read_parquet(log_path)
        missing = LOG_COLUMNS.difference(logs.columns)
        if missing:
            raise ValueError(f"{log_path} missing columns: {sorted(missing)}")
        logs = logs.groupby(["window_id", "service"], as_index=False).agg(
            log_count=("log_count", "sum"),
            error_count=("error_count", "sum"),
            warning_count=("warning_count", "sum"),
        )
    else:
        logs = pd.DataFrame(columns=["window_id", "service", "log_count", "error_count", "warning_count"])

    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    for window_id, edges in traces.groupby("window_id", sort=True):
        services = sorted(set(edges["caller"]) | set(edges["callee"]))
        node = pd.DataFrame({"service": services})
        node["node_id"] = range(len(node))
        node["case"] = case
        node["window_id"] = int(window_id)
        edge_meta = edges.iloc[0]
        node["window_start"] = int(edge_meta["window_start"])
        node["window_end"] = int(edge_meta["window_end"])
        node["relative_start"] = int(edge_meta["relative_start"])
        node["relative_end"] = int(edge_meta["relative_end"])
        node["phase"] = str(edge_meta["phase"])
        node["anomaly_label"] = int(edge_meta["anomaly_label"])
        node = node.merge(logs.loc[logs["window_id"] == window_id], on=["window_id", "service"], how="left")
        for column in ("log_count", "error_count", "warning_count"):
            node[column] = node[column].fillna(0).astype("int64")
        for column in ("window_start", "window_end", "relative_start", "relative_end"):
            node[column] = node[column].fillna(int(edge_meta[column])).astype("int64")
        node["phase"] = node["phase"].fillna(str(edge_meta["phase"]))
        node["anomaly_label"] = node["anomaly_label"].fillna(int(edge_meta["anomaly_label"])).astype("int8")
        node_frames.append(node)
        lookup = dict(zip(node["service"], node["node_id"]))
        edge = edges[["case", "window_id", "window_start", "window_end", "relative_start", "relative_end", "phase", "anomaly_label", "caller", "callee", "call_count", "error_count", "root_cause_service", "fault"]].copy()
        edge["src"] = edge["caller"].map(lookup).astype("int64")
        edge["dst"] = edge["callee"].map(lookup).astype("int64")
        edge_frames.append(edge)

    nodes = pd.concat(node_frames, ignore_index=True)
    edges = pd.concat(edge_frames, ignore_index=True)
    destination = output_root / case
    destination.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(destination / "nodes.parquet", index=False)
    edges.to_parquet(destination / "edges.parquet", index=False)
    return {
        "case": case,
        "trace_source": str(trace_path),
        "trace_source_sha256": sha256(trace_path),
        "log_source": str(log_path) if log_path.exists() else None,
        "windows": int(traces["window_id"].nunique()),
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "services": int(nodes["service"].nunique()),
        "output_nodes": str((destination / "nodes.parquet")),
        "output_edges": str((destination / "edges.parquet")),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    width = args.window_seconds or int(config["preprocessing"]["window_seconds"])
    if width != 60:
        raise ValueError("Phase 7 currently builds the cached default 60-second graph snapshots")
    processed = root / config["data"]["processed_dir"]
    index_path = root / config["data"]["cases_index"]
    index = pd.read_parquet(index_path)
    split = make_case_split(index, int(config.get("seed", 42)))
    split_dir = processed / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "re2tt_case_split.parquet"
    split.to_parquet(split_path, index=False)
    split_meta = {
        "dataset": "RE2-TT", "seed": int(config.get("seed", 42)),
        "grouping": "system + root_cause_service + fault",
        "group_count": int(split["split_group"].nunique()),
        "case_counts": {str(k): int(v) for k, v in split["split"].value_counts().sort_index().items()},
        "group_counts": {str(k): int(v) for k, v in split.groupby("split")["split_group"].nunique().items()},
        "source": str(index_path.relative_to(root)),
        "output": str(split_path.relative_to(root)),
    }
    (split_dir / "re2tt_case_split.json").write_text(json.dumps(split_meta, indent=2) + "\n", encoding="utf-8")

    graph_root = processed / "graphs" / "re2tt" / "60s"
    graph_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures: dict[str, str] = {}
    for case in split["case"]:
        try:
            summary = build_case(str(case), processed / "windows", graph_root)
            for key in ("trace_source", "log_source", "output_nodes", "output_edges"):
                if summary.get(key):
                    summary[key] = str(Path(str(summary[key])).relative_to(root))
            summaries.append(summary)
        except Exception as exc:
            failures[str(case)] = f"{type(exc).__name__}: {exc}"
            print(f"failed {case}: {exc}", flush=True)
    manifest = {
        "dataset": "RE2-TT", "window_seconds": 60, "seed": int(config.get("seed", 42)),
        "node_schema": ["case", "window_id", "node_id", "service", "log_count", "error_count", "warning_count", "phase", "anomaly_label"],
        "edge_schema": ["case", "window_id", "src", "dst", "caller", "callee", "call_count", "error_count", "phase", "anomaly_label"],
        "cases_requested": int(len(split)), "cases_completed": int(len(summaries)),
        "cases_failed": failures, "graphs": summaries,
        "split": split_meta,
        "config_file": str((root / args.config).relative_to(root)),
    }
    manifest_path = graph_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "cases_requested": len(split), "cases_completed": len(summaries),
        "cases_failed": len(failures), "total_nodes": sum(int(x["nodes"]) for x in summaries),
        "total_edges": sum(int(x["edges"]) for x in summaries),
        "split": split_meta, "manifest": str(manifest_path.relative_to(root)),
    }, indent=2))


if __name__ == "__main__":
    main()
