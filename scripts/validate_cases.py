"""Validate the RCAEval cases index without reading telemetry files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "raw" / "cases.parquet"
    output = root / "data" / "processed" / "cases_schema_validation.json"
    frame = pd.read_parquet(source)

    time_invalid = (frame["inject_time"] < frame["time_start"]) | (
        frame["inject_time"] > frame["time_end"]
    )
    result = {
        "source": "phamquiluan/RCAEval/cases.parquet",
        "source_file": str(source.relative_to(root)),
        "shape": {"rows": int(len(frame)), "columns": int(len(frame.columns))},
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "null_counts": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if int(count)
        },
        "duplicate_case_ids": int(frame["case"].duplicated().sum()),
        "dataset_counts": {
            str(key): int(value)
            for key, value in frame["dataset"].value_counts().sort_index().items()
        },
        "re2_tt": {
            "n_cases": int((frame["dataset"] == "RE2-TT").sum()),
            "fault_counts": {
                str(key): int(value)
                for key, value in frame.loc[frame["dataset"] == "RE2-TT", "fault"]
                .value_counts()
                .sort_index()
                .items()
            },
            "root_cause_counts": {
                str(key): int(value)
                for key, value in frame.loc[
                    frame["dataset"] == "RE2-TT", "root_cause_service"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "consistency_checks": {
            "negative_count_violations": {
                column: int((frame[column] < 0).sum())
                for column in [
                    "n_metrics",
                    "n_timesteps",
                    "normal_timesteps",
                    "faulty_timesteps",
                    "n_logs",
                    "n_traces",
                ]
            },
            "availability_violations": {
                "has_logs_false_but_positive_count": int(
                    ((~frame["has_logs"]) & (frame["n_logs"] > 0)).sum()
                ),
                "has_traces_false_but_positive_count": int(
                    ((~frame["has_traces"]) & (frame["n_traces"] > 0)).sum()
                ),
            },
            "time_boundary_violations": int(time_invalid.sum()),
        },
        "time_boundary_cases": frame.loc[
            time_invalid,
            [
                "case",
                "dataset",
                "inject_time",
                "time_start",
                "time_end",
            ],
        ].to_dict(orient="records"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
