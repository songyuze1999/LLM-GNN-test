"""Cache frozen Qwen embeddings for preprocessed RE2-TT log windows.

The cache is written one failure case at a time so an interrupted run can be
resumed without recomputing completed cases. Embeddings are always stored as
float32 CPU arrays, while the encoder uses the configured inference dtype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.llm_encoder import FrozenQwenEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--window-seconds", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    preprocessing = config["preprocessing"]
    width = int(args.window_seconds or preprocessing["window_seconds"])
    supported = {int(value) for value in preprocessing["supported_window_seconds"]}
    if width not in supported:
        raise ValueError(f"window must be one of {sorted(supported)}, got {width}")
    batch_size = int(args.batch_size or config["llm"].get("batch_size", 4))
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    processed = root / config["data"]["processed_dir"]
    source_root = processed / "windows"
    output_root = processed / "embeddings" / "qwen" / f"{width}s"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "dataset": config["data"].get("dataset", "RE2-TT"),
            "window_seconds": width,
            "seed": seed,
            "config_file": str(config_path.relative_to(root)),
            "source_root": str(source_root.relative_to(root)),
            "cases_requested": [],
            "cases_completed": [],
            "cases_missing_logs": [],
            "cases_failed": {},
        }

    index = pd.read_parquet(root / config["data"]["cases_index"])
    dataset = str(config["data"].get("dataset", "RE2-TT"))
    cases = index.loc[index["dataset"].eq(dataset), "case"].astype(str).sort_values().tolist()
    manifest["cases_requested"] = cases
    completed = set(manifest.get("cases_completed", []))
    missing = set(manifest.get("cases_missing_logs", []))
    failed = dict(manifest.get("cases_failed", {}))
    pending = [case for case in cases if case not in completed and case not in missing]
    if args.max_cases is not None:
        pending = pending[: max(0, int(args.max_cases))]

    llm = config["llm"]
    encoder = FrozenQwenEncoder(
        llm["model_name"],
        load_path=str(root / llm["local_path"]) if llm.get("local_path") else None,
        revision=llm.get("revision", "main"),
        pooling=llm.get("pooling", "mean"),
        max_length=int(llm.get("max_length", 512)),
        dtype=llm.get("dtype", "bfloat16"),
        device=args.device,
    )
    manifest["encoder"] = {key: value for key, value in encoder.metadata().__dict__.items()}
    manifest["batch_size"] = batch_size
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    for position, case in enumerate(pending, start=1):
        source = source_root / case / f"{width}s" / "log_windows.parquet"
        if not source.exists():
            missing.add(case)
            print(f"[{position}/{len(pending)}] missing logs: {case}", flush=True)
            manifest["cases_missing_logs"] = sorted(missing)
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            continue
        try:
            columns = [
                "case", "window_id", "window_start", "window_end", "relative_start",
                "relative_end", "phase", "anomaly_label", "service", "log_count",
                "error_count", "warning_count", "root_cause_service", "fault", "messages",
            ]
            logs = pd.read_parquet(source, columns=columns)
            embeddings = encoder.encode(logs["messages"].fillna("").astype(str).tolist(), batch_size=batch_size)
            array_path = output_root / f"{case}.npy"
            index_path = output_root / f"{case}.parquet"
            metadata_path = output_root / f"{case}.json"
            np.save(array_path, embeddings.numpy())
            logs.drop(columns=["messages"]).to_parquet(index_path, index=False)
            metadata = {
                "case": case,
                "source": str(source.relative_to(root)),
                "source_sha256": sha256(source),
                "rows": int(len(logs)),
                "embedding_shape": list(embeddings.shape),
                "embedding_dtype": str(embeddings.numpy().dtype),
                "encoder": manifest["encoder"],
                "config_file": manifest["config_file"],
                "seed": seed,
                "array_file": str(array_path.relative_to(root)),
                "index_file": str(index_path.relative_to(root)),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            completed.add(case)
            failed.pop(case, None)
            print(f"[{position}/{len(pending)}] cached {case}: rows={len(logs)}", flush=True)
        except Exception as exc:  # preserve the failure for a later retry
            failed[case] = f"{type(exc).__name__}: {exc}"
            print(f"[{position}/{len(pending)}] failed {case}: {exc}", flush=True)
        manifest["cases_completed"] = sorted(completed)
        manifest["cases_missing_logs"] = sorted(missing)
        manifest["cases_failed"] = failed
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    manifest["cases_completed"] = sorted(completed)
    manifest["cases_missing_logs"] = sorted(missing)
    manifest["cases_failed"] = failed
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "cases_requested": len(cases),
        "cases_completed": len(completed),
        "cases_missing_logs": len(missing),
        "cases_failed": len(failed),
        "manifest": str(manifest_path.relative_to(root)),
    }, indent=2))


if __name__ == "__main__":
    main()
