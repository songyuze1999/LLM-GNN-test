"""Validate frozen Qwen embeddings on a small, reproducible log sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.llm_encoder import FrozenQwenEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default="data/processed/windows/re2tt_ts-auth-service_cpu_2/60s/log_windows.parquet")
    parser.add_argument("--sample-size", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    source = root / args.source
    if not source.exists():
        raise FileNotFoundError(source)
    logs = pd.read_parquet(source, columns=["case", "window_id", "service", "messages", "phase", "anomaly_label"])
    half = args.sample_size // 2
    normal = logs.loc[logs["phase"] == "normal"].head(half)
    faulty = logs.loc[logs["phase"] != "normal"].head(args.sample_size - len(normal))
    selected = list(normal.index) + list(faulty.index)
    selected.extend(index for index in logs.index if index not in selected)
    sample = logs.loc[selected[: args.sample_size]].copy()
    if len(sample) < 10 or len(sample) > 100:
        raise ValueError(f"sample must contain 10-100 logs, got {len(sample)}")
    llm = config["llm"]
    encoder = FrozenQwenEncoder(
        llm["model_name"],
        load_path=str(root / llm["local_path"]) if llm.get("local_path") else None,
        revision=llm.get("revision", "main"),
        pooling=llm.get("pooling", "mean"),
        max_length=int(llm.get("max_length", 512)),
        dtype=llm.get("dtype", "bfloat16"),
    )
    embeddings = encoder.encode(sample["messages"].fillna("").tolist(), batch_size=int(llm.get("batch_size", 4)))
    output_dir = root / "data" / "processed" / "embeddings" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem.replace("_windows", "")
    embedding_path = output_dir / f"{stem}_qwen.npy"
    metadata_path = output_dir / f"{stem}_qwen.json"
    np.save(embedding_path, embeddings.numpy())
    metadata = {
        "source": str(source.relative_to(root)),
        "source_sha256": sha256(source),
        "sample_rows": int(len(sample)),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.numpy().dtype),
        "phase_counts": {str(key): int(value) for key, value in sample["phase"].value_counts().items()},
        "encoder": {key: value for key, value in encoder.metadata().__dict__.items()},
        "config_file": str((root / args.config).relative_to(root)),
        "seed": int(config.get("seed", 42)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    print(f"embedding_file={embedding_path}")
    print(f"metadata_file={metadata_path}")


if __name__ == "__main__":
    main()
