"""Preprocess all locally downloaded RE2-TT cases sequentially."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.window_builder import build_log_windows, build_metric_windows, build_trace_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--window-seconds", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    width = args.window_seconds or int(config["preprocessing"]["window_seconds"])
    supported = set(config["preprocessing"]["supported_window_seconds"])
    if width not in supported:
        raise ValueError(f"window must be one of {sorted(supported)}, got {width}")
    raw_dir = root / config["data"]["raw_dir"]
    output_root = root / config["data"]["processed_dir"] / "windows"
    index = pd.read_parquet(root / config["data"]["cases_index"])
    index = index.loc[index["dataset"] == config["data"]["dataset"]].sort_values("case")
    manifest: dict[str, object] = {
        "dataset": config["data"]["dataset"],
        "window_seconds": width,
        "cases_requested": int(len(index)),
        "cases_completed": [],
        "cases_missing_logs": [],
        "cases_failed": {},
    }
    for row in index.itertuples(index=False):
        case = str(row.case)
        try:
            case_dir = raw_dir / case
            metrics_path = case_dir / "metrics.parquet"
            inject_path = case_dir / "inject_time.txt"
            if not metrics_path.exists() or not inject_path.exists():
                raise FileNotFoundError(f"missing metrics or inject_time for {case}")
            inject_time = int(inject_path.read_text(encoding="utf-8").strip())
            if inject_time < int(row.time_start) or inject_time > int(row.time_end):
                raise ValueError(f"invalid injection boundary for {case}")
            metric_times = pd.read_parquet(metrics_path, columns=["time"])
            origin = int(pd.to_numeric(metric_times["time"], errors="coerce").min())
            destination = output_root / case / f"{width}s"
            destination.mkdir(parents=True, exist_ok=True)
            metric_windows = build_metric_windows(
                metrics_path,
                origin=origin,
                width=width,
                inject_time=inject_time,
                case=case,
                root_cause_service=str(row.root_cause_service),
                fault=str(row.fault),
            )
            metric_windows.to_parquet(destination / "metric_windows.parquet", index=False)
            logs_path = case_dir / "logs.parquet"
            if logs_path.exists():
                window_index, log_windows = build_log_windows(
                    logs_path,
                    origin=origin,
                    width=width,
                    inject_time=inject_time,
                    case=case,
                    root_cause_service=str(row.root_cause_service),
                    fault=str(row.fault),
                    keep_messages=bool(config["preprocessing"]["keep_messages"]),
                    max_message_chars=int(config["preprocessing"]["max_message_chars"]),
                )
                window_index.to_parquet(destination / "window_index.parquet", index=False)
                log_windows.to_parquet(destination / "log_windows.parquet", index=False)
            else:
                manifest["cases_missing_logs"].append(case)
            traces_path = case_dir / "traces.parquet"
            if traces_path.exists():
                trace_edges = build_trace_edges(
                    traces_path,
                    origin=origin,
                    width=width,
                    inject_time=inject_time,
                    case=case,
                    root_cause_service=str(row.root_cause_service),
                    fault=str(row.fault),
                )
                trace_edges.to_parquet(destination / "trace_edges.parquet", index=False)
            manifest["cases_completed"].append(case)
            print(f"completed {case}", flush=True)
        except Exception as exc:  # preserve the case and continue the batch
            manifest["cases_failed"][case] = f"{type(exc).__name__}: {exc}"
            print(f"failed {case}: {exc}", flush=True)
    manifest["cases_completed"] = sorted(manifest["cases_completed"])
    manifest["cases_missing_logs"] = sorted(manifest["cases_missing_logs"])
    manifest_path = root / config["data"]["processed_dir"] / "re2tt_preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "cases_requested": manifest["cases_requested"],
        "cases_completed": len(manifest["cases_completed"]),
        "cases_missing_logs": len(manifest["cases_missing_logs"]),
        "cases_failed": len(manifest["cases_failed"]),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
