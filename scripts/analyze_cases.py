"""Summarize the official RCAEval cases index without downloading telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "raw" / "cases.parquet"
    output = root / "data" / "processed" / "dataset_statistics.json"
    frame = pd.read_parquet(source)

    # RCAEval encodes the three target systems as ob, ss, and tt.
    re2 = frame[frame["dataset"].astype(str).str.startswith("RE2-")].copy()
    re2_tt = frame[frame["dataset"].eq("RE2-TT")].copy()

    def counts(series: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}

    def summary(subset: pd.DataFrame) -> dict[str, object]:
        return {
            "n_cases": int(len(subset)),
            "systems": counts(subset["system_name"]),
            "fault_types": counts(subset["fault"]),
            "root_cause_services": counts(subset["root_cause_service"]),
            "has_logs": counts(subset["has_logs"]),
            "has_traces": counts(subset["has_traces"]),
            "has_root_cause_file": counts(subset["has_root_cause_file"]),
            "n_logs": {
                "total": int(subset["n_logs"].sum()),
                "min": int(subset["n_logs"].min()) if len(subset) else 0,
                "max": int(subset["n_logs"].max()) if len(subset) else 0,
            },
            "n_traces": {
                "total": int(subset["n_traces"].sum()),
                "min": int(subset["n_traces"].min()) if len(subset) else 0,
                "max": int(subset["n_traces"].max()) if len(subset) else 0,
            },
            "n_metrics": {
                "total": int(subset["n_metrics"].sum()),
                "min": int(subset["n_metrics"].min()) if len(subset) else 0,
                "max": int(subset["n_metrics"].max()) if len(subset) else 0,
            },
            "time_range": {
                "min": int(subset["time_start"].min()) if len(subset) else None,
                "max": int(subset["time_end"].max()) if len(subset) else None,
            },
        }

    result = {
        "source": "phamquiluan/RCAEval/cases.parquet",
        "source_file": str(source.relative_to(root)),
        "total_cases": int(len(frame)),
        "datasets": counts(frame["dataset"]),
        "systems": counts(frame["system_name"]),
        "fault_types": counts(frame["fault"]),
        "re2": summary(re2),
        "re2_tt": summary(re2_tt),
        "re2_tt_cases": re2_tt[["case", "system_name", "root_cause_service", "fault", "inject_time", "has_logs", "n_logs", "has_traces", "n_traces"]].to_dict(orient="records"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
