"""Create reproducible time-windowed tables for locally available RCAEval cases."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

# Keep the documented ``python scripts/...`` entry point independent of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.window_builder import build_log_windows, build_metric_windows, build_trace_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--case", required=True)
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
    output = root / config["data"]["processed_dir"] / "windows" / args.case / f"{width}s"
    index = pd.read_parquet(root / config["data"]["cases_index"])
    matches = index.loc[index["case"] == args.case]
    if len(matches) != 1:
        raise ValueError(f"case must occur exactly once in the index: {args.case}")
    row = matches.iloc[0]
    case_dir = raw_dir / args.case
    if not case_dir.is_dir():
        raise FileNotFoundError(f"telemetry directory is not present: {case_dir}")
    inject_time = int((case_dir / "inject_time.txt").read_text(encoding="utf-8").strip())
    if inject_time < int(row.time_start) or inject_time > int(row.time_end):
        raise ValueError(f"case has an invalid injection boundary in the index: {args.case}")
    tables: dict[str, pd.DataFrame] = {}
    metrics = pd.read_parquet(case_dir / "metrics.parquet", columns=["time"])
    origin = int(pd.to_numeric(metrics["time"], errors="coerce").min())
    logs_path = case_dir / "logs.parquet"
    if logs_path.exists():
        window_meta, tables["log_windows"] = build_log_windows(
            logs_path,
            origin=origin,
            width=width,
            inject_time=inject_time,
            case=args.case,
            root_cause_service=str(row.root_cause_service),
            fault=str(row.fault),
            keep_messages=bool(config["preprocessing"]["keep_messages"]),
            max_message_chars=int(config["preprocessing"]["max_message_chars"]),
        )
        tables["window_index"] = window_meta
    tables["metric_windows"] = build_metric_windows(
        case_dir / "metrics.parquet",
        origin=origin,
        width=width,
        inject_time=inject_time,
        case=args.case,
        root_cause_service=str(row.root_cause_service),
        fault=str(row.fault),
    )
    traces_path = case_dir / "traces.parquet"
    if traces_path.exists():
        tables["trace_edges"] = build_trace_edges(
            traces_path,
            origin=origin,
            width=width,
            inject_time=inject_time,
            case=args.case,
            root_cause_service=str(row.root_cause_service),
            fault=str(row.fault),
        )
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(output / f"{name}.parquet", index=False)
    print(f"case={args.case} window_seconds={width} origin={origin} inject_time={inject_time}")
    for name, table in tables.items():
        print(f"{name}: rows={len(table)} path={output / (name + '.parquet')}")


if __name__ == "__main__":
    main()
