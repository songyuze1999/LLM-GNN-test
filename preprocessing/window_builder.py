"""Build case-level, time-windowed RCAEval tables without resampling telemetry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _phase(start: int, end: int, inject_time: int) -> str:
    if end < inject_time:
        return "normal"
    if start >= inject_time:
        return "faulty"
    return "mixed"


def _window_columns(frame: pd.DataFrame, timestamp: str, origin: int, width: int) -> pd.DataFrame:
    values = pd.to_numeric(frame[timestamp], errors="coerce")
    frame = frame.loc[values.notna()].copy()
    frame["_timestamp"] = values.loc[frame.index].astype("int64")
    frame["window_id"] = ((frame["_timestamp"] - origin) // width).astype("int64")
    frame["window_start"] = origin + frame["window_id"] * width
    frame["window_end"] = frame["window_start"] + width - 1
    return frame


def _window_meta(frame: pd.DataFrame, inject_time: int) -> pd.DataFrame:
    grouped = frame.groupby(["window_id", "window_start", "window_end"], sort=True)
    result = grouped.size().rename("event_count").reset_index()
    result["relative_start"] = result["window_start"] - inject_time
    result["relative_end"] = result["window_end"] - inject_time
    result["phase"] = [
        _phase(int(start), int(end), inject_time)
        for start, end in zip(result["window_start"], result["window_end"])
    ]
    result["anomaly_label"] = (result["phase"] != "normal").astype("int8")
    return result


def build_log_windows(
    path: Path,
    *,
    origin: int,
    width: int,
    inject_time: int,
    case: str,
    root_cause_service: str,
    fault: str,
    keep_messages: bool = True,
    max_message_chars: int = 2_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["timestamp", "container_name", "message"]
    logs = pd.read_parquet(path, columns=columns)
    logs = _window_columns(logs, "timestamp", origin, width)
    if logs.empty:
        return pd.DataFrame(), pd.DataFrame()
    meta = _window_meta(logs, inject_time)
    grouped = logs.groupby(["window_id", "window_start", "window_end", "container_name"], sort=True)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        window_id, window_start, window_end, service = keys
        messages = "\n".join(group["message"].dropna().astype(str)) if keep_messages else ""
        rows.append(
            {
                "case": case,
                "window_id": int(window_id),
                "window_start": int(window_start),
                "window_end": int(window_end),
                "relative_start": int(window_start - inject_time),
                "relative_end": int(window_end - inject_time),
                "phase": _phase(int(window_start), int(window_end), inject_time),
                "anomaly_label": int(window_end >= inject_time),
                "service": str(service),
                "log_count": int(len(group)),
                "error_count": int(group["message"].astype(str).str.contains(r"\bERROR\b|\bEXCEPTION\b", case=False, regex=True).sum()),
                "warning_count": int(group["message"].astype(str).str.contains(r"\bWARN(?:ING)?\b", case=False, regex=True).sum()),
                "messages": messages[:max_message_chars],
                "root_cause_service": root_cause_service,
                "fault": fault,
            }
        )
    return meta.assign(case=case, root_cause_service=root_cause_service, fault=fault), pd.DataFrame(rows)


def build_metric_windows(
    path: Path,
    *,
    origin: int,
    width: int,
    inject_time: int,
    case: str,
    root_cause_service: str,
    fault: str,
) -> pd.DataFrame:
    metrics = pd.read_parquet(path)
    metrics = _window_columns(metrics, "time", origin, width)
    if metrics.empty:
        return pd.DataFrame()
    metric_columns = [column for column in metrics.columns if column not in {"time", "_timestamp", "window_id", "window_start", "window_end"}]
    long = metrics.melt(
        id_vars=["window_id", "window_start", "window_end"],
        value_vars=metric_columns,
        var_name="metric_name",
        value_name="value",
    )
    split = long["metric_name"].astype(str).str.rsplit("_", n=1, expand=True)
    long["service"] = split[0]
    long["metric"] = split[1]
    aggregate = long.groupby(["window_id", "window_start", "window_end", "service", "metric"], sort=True)["value"].agg(
        mean="mean", min="min", max="max", non_null="count"
    ).reset_index()
    aggregate["case"] = case
    aggregate["relative_start"] = aggregate["window_start"] - inject_time
    aggregate["relative_end"] = aggregate["window_end"] - inject_time
    aggregate["phase"] = [
        _phase(int(start), int(end), inject_time)
        for start, end in zip(aggregate["window_start"], aggregate["window_end"])
    ]
    aggregate["anomaly_label"] = (aggregate["phase"] != "normal").astype("int8")
    aggregate["root_cause_service"] = root_cause_service
    aggregate["fault"] = fault
    return aggregate


def build_trace_edges(
    path: Path,
    *,
    origin: int,
    width: int,
    inject_time: int,
    case: str,
    root_cause_service: str,
    fault: str,
) -> pd.DataFrame:
    columns = ["traceID", "spanID", "serviceName", "parentSpanID", "startTimeMillis", "duration", "statusCode"]
    traces = pd.read_parquet(path, columns=columns)
    traces["_timestamp"] = pd.to_numeric(traces["startTimeMillis"], errors="coerce").floordiv(1000)
    traces = traces.loc[traces["_timestamp"].notna()].copy()
    if traces.empty:
        return pd.DataFrame()
    service_by_span = traces.dropna(subset=["spanID"]).set_index("spanID")["serviceName"].to_dict()
    traces["caller"] = traces["parentSpanID"].map(service_by_span)
    traces = traces.loc[traces["caller"].notna() & traces["serviceName"].notna()]
    traces = traces.loc[traces["caller"] != traces["serviceName"]].copy()
    if traces.empty:
        return pd.DataFrame()
    traces["window_id"] = ((traces["_timestamp"] - origin) // width).astype("int64")
    traces["window_start"] = origin + traces["window_id"] * width
    traces["window_end"] = traces["window_start"] + width - 1
    grouped = traces.groupby(["window_id", "window_start", "window_end", "caller", "serviceName"], sort=True)
    result = grouped.agg(
        call_count=("traceID", "size"),
        error_count=("statusCode", lambda values: int(values.notna().sum() and (pd.to_numeric(values, errors="coerce") >= 400).sum())),
    ).reset_index().rename(columns={"serviceName": "callee"})
    result["case"] = case
    result["relative_start"] = result["window_start"] - inject_time
    result["relative_end"] = result["window_end"] - inject_time
    result["phase"] = [
        _phase(int(start), int(end), inject_time)
        for start, end in zip(result["window_start"], result["window_end"])
    ]
    result["anomaly_label"] = (result["phase"] != "normal").astype("int8")
    result["root_cause_service"] = root_cause_service
    result["fault"] = fault
    return result
