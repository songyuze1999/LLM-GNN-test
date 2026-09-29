"""Run Phase 8 anomaly-detection baselines on case-level splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def metric_dict(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    pred = (score >= threshold).astype("int8")
    result: dict[str, float | None] = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, score)),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else None
    return result


def load_text_samples(root: Path, split: pd.DataFrame, max_chars: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_row in split.itertuples(index=False):
        path = root / "data/processed/windows" / str(case_row.case) / "60s/log_windows.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["case", "window_id", "messages", "anomaly_label", "phase"])
        grouped = frame.groupby(["case", "window_id"], sort=True).agg(
            text=("messages", lambda values: "\n".join(values.fillna("").astype(str))[:max_chars]),
            anomaly_label=("anomaly_label", "max"),
            phase=("phase", "first"),
        ).reset_index()
        grouped["split"] = str(case_row.split)
        rows.extend(grouped.to_dict(orient="records"))
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("no log windows available for TF-IDF baseline")
    return result


def load_qwen_samples(root: Path, split: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    for case_row in split.itertuples(index=False):
        stem = root / "data/processed/embeddings/qwen/60s" / str(case_row.case)
        index_path, array_path = stem.with_suffix(".parquet"), stem.with_suffix(".npy")
        if not index_path.exists() or not array_path.exists():
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
    return pd.DataFrame(rows), np.stack(vectors)


def fit_and_score(name: str, model: object, train_x, train_y, valid_x, valid_y, test_x, test_y) -> tuple[dict[str, object], np.ndarray]:
    started = time.perf_counter()
    model.fit(train_x, train_y)
    valid_score = model.predict_proba(valid_x)[:, 1]
    test_score = model.predict_proba(test_x)[:, 1]
    return {
        "model": name,
        "fit_seconds": round(time.perf_counter() - started, 3),
        "validation": metric_dict(valid_y, valid_score),
        "test": metric_dict(test_y, test_score),
    }, test_score


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    split = pd.read_parquet(root / "data/processed/splits/re2tt_case_split.parquet")
    baseline = config["baselines"]
    output = root / "results" / "baselines"
    output.mkdir(parents=True, exist_ok=True)

    text = load_text_samples(root, split, int(baseline["tfidf_max_chars_per_window"]))
    train, valid, test = (text[text.split == part].reset_index(drop=True) for part in ("train", "validation", "test"))
    vectorizer = TfidfVectorizer(
        max_features=int(baseline["tfidf_max_features"]), min_df=int(baseline["tfidf_min_df"]),
        ngram_range=tuple(int(x) for x in baseline["tfidf_ngram_range"]), sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train.text)
    valid_matrix = vectorizer.transform(valid.text)
    test_matrix = vectorizer.transform(test.text)
    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for name, classifier in (
        ("tfidf_logistic_regression", LogisticRegression(max_iter=500, random_state=seed, class_weight="balanced")),
        ("tfidf_mlp", MLPClassifier(hidden_layer_sizes=(int(baseline["mlp_hidden_size"]),), max_iter=int(baseline["mlp_max_iter"]), early_stopping=bool(baseline["mlp_early_stopping"]), random_state=seed)),
    ):
        metrics, score = fit_and_score(name, classifier, train_matrix, train.anomaly_label, valid_matrix, valid.anomaly_label, test_matrix, test.anomaly_label)
        results.append(metrics)
        predictions.append(pd.DataFrame({"model": name, "case": test.case, "window_id": test.window_id, "label": test.anomaly_label, "score": score}))

    qwen, qwen_vectors = load_qwen_samples(root, split)
    qtrain, qvalid, qtest = (qwen[qwen.split == part].reset_index(drop=True) for part in ("train", "validation", "test"))
    qtrain_idx = qwen.index[qwen.split == "train"].to_numpy()
    qvalid_idx = qwen.index[qwen.split == "validation"].to_numpy()
    qtest_idx = qwen.index[qwen.split == "test"].to_numpy()
    qmodel = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(int(baseline["mlp_hidden_size"]),), max_iter=int(baseline["mlp_max_iter"]), early_stopping=bool(baseline["mlp_early_stopping"]), random_state=seed)),
    ])
    metrics, score = fit_and_score("qwen_embedding_mlp", qmodel, qwen_vectors[qtrain_idx], qtrain.anomaly_label, qwen_vectors[qvalid_idx], qvalid.anomaly_label, qwen_vectors[qtest_idx], qtest.anomaly_label)
    results.append(metrics)
    predictions.append(pd.DataFrame({"model": "qwen_embedding_mlp", "case": qtest.case, "window_id": qtest.window_id, "label": qtest.anomaly_label, "score": score}))

    payload = {"dataset": "RE2-TT", "seed": seed, "window_seconds": 60, "text_cases": int(text.case.nunique()), "qwen_cases": int(qwen.case.nunique()), "tfidf_features": int(train_matrix.shape[1]), "models": results, "config_file": str((root / args.config).relative_to(root))}
    (output / "phase8_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(output / "phase8_test_predictions.csv", index=False)
    (output / "phase8_run_metadata.json").write_text(json.dumps({"seed": seed, "split_file": "data/processed/splits/re2tt_case_split.parquet", "qwen_pooling": "mean over services per window", "missing_log_cases": sorted(set(split.case) - set(text.case)), "missing_embedding_cases": sorted(set(split.case) - set(qwen.case))}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
