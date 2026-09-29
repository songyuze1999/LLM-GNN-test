#!/usr/bin/env bash
set -euo pipefail

# Run the reproducible RE2-TT 60-second pipeline from the repository root.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/default.yaml}"

echo "[1/9] validate cases"
"$PYTHON_BIN" scripts/validate_cases.py
echo "[2/9] analyze dataset index"
"$PYTHON_BIN" scripts/analyze_cases.py
echo "[3/9] preprocess locally available RE2-TT telemetry"
"$PYTHON_BIN" scripts/preprocess_re2tt.py --config "$CONFIG"
echo "[4/9] validate the 10-100 log embedding sample"
"$PYTHON_BIN" scripts/validate_qwen_embeddings.py --config "$CONFIG"
echo "[5/9] cache frozen Qwen embeddings (completed cases are skipped)"
"$PYTHON_BIN" scripts/cache_qwen_embeddings.py --config "$CONFIG"
echo "[6/9] build service graphs and the case-level split"
"$PYTHON_BIN" scripts/build_service_graphs.py --config "$CONFIG"
echo "[7/9] run TF-IDF and Qwen baselines"
"$PYTHON_BIN" scripts/run_baselines.py --config "$CONFIG"
echo "[8/9] run the dedicated Qwen + MLP experiment"
"$PYTHON_BIN" scripts/run_qwen_mlp.py --config "$CONFIG"
echo "[9/9] run GCN, GAT, GATv2 and available follow-up analyses"
"$PYTHON_BIN" scripts/run_graph_experiments.py --config "$CONFIG"
echo "Reproduction completed. See results/ and migration/HANDOFF.md."
