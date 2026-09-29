# LLM-GNN-RCA Project Rules

## Project objective

This repository implements the master's thesis experiments for **microservice log anomaly detection and root-cause localization using Qwen2.5-7B-Instruct and GATv2**. The technical route is fixed: the frozen LLM understands log semantics; GNN/GATv2 models service-call structure and anomaly propagation.

## Non-negotiable data and reproducibility rules

- Do not fabricate RCAEval data, labels, service dependencies, predictions, or results.
- Do not modify source dataset labels or telemetry.
- Do not use telemetry from a test failure case for training or model selection.
- Split at failure-case level; windows from one case must never cross train/validation/test splits. Consider related duplicate injections when making splits.
- Use fixed seed 42 for Python, NumPy, PyTorch, and CUDA. Save every split under `data/processed/splits/`.
- Keep all experiment parameters in configuration files, primarily `configs/default.yaml`; do not hard-code experimental settings.
- Cache Qwen embeddings and record the model revision/configuration used to produce them.
- Preserve raw data, checkpoints, predictions, results, and logs. Do not delete them without explicit user instruction.
- If a data format or dependency relationship is uncertain, inspect the source files and stop the affected step rather than guessing.

## Environment rules

- Prefer the Conda environment `llm-gnn`; create it if absent.
- Current server baseline (verified 2026-09-27): Ubuntu 24.04.4 LTS, Python 3.12.14, NVIDIA GeForce RTX 5090 32 GB, NVIDIA driver 580.105.08. The driver reports CUDA 13.0 capability.
- The `llm-gnn` environment uses the official PyTorch `2.9.1+cu130` wheel and its CUDA 13.0 runtime packages, with `torchvision 0.24.1+cu130`, `torchaudio 2.9.1+cu130`, and PyTorch Geometric `2.8.0.post1` plus the matching `pt29cu130` extension wheels.
- The system `nvcc` currently reports CUDA toolkit 12.8 while the PyTorch runtime is CUDA 13.0. Keep this distinction explicit: ordinary PyTorch execution uses the wheel runtime; CUDA extensions must be checked against the active toolkit before compiling.
- Reuse the existing CUDA, NVIDIA driver, PyTorch, and PyG installation when available. Never reinstall, downgrade, or change the NVIDIA driver or system CUDA toolkit. Do not change the pinned PyTorch/PyG versions without explicit instruction.
- Support CUDA GPU execution and CPU fallback. Prefer BF16 for frozen Qwen inference when the device supports it.
- Do not use absolute paths in project code. Resolve paths from the repository root/configuration.

## Dataset workflow

- Use the official RCAEval source and its Hugging Face Parquet representation (`phamquiluan/RCAEval`) when available.
- Start with `cases.parquet`; inspect `case`, `dataset`, `system`, `root_cause_service`, `fault`, `injection_time`, and telemetry availability/size.
- Phase 1 downloads/analyzes only the RE2-TT index and the minimum files needed to understand its schema. Do not download all 735 cases or telemetry at once.
- Prioritize RE2-TT, then RE2-OB and RE2-SS, and only later consider RE3.
- Build service edges only from reliable caller/callee information in traces. Never invent graph edges.
- Use configurable windows (default 60 seconds; support 30 and 120 seconds), ordered by timestamp and labeled relative to each case's injection time.

## Required model route

- `models/llm_encoder.py` must expose frozen Qwen hidden-state embeddings (mean pooling by default, with pooling alternatives). Do not use generated answers as the primary feature.
- Graph nodes are services, not individual log lines. Node features include Qwen log embeddings and log/error/warning counts and ratios, with optional metrics features.
- Implement GCN, GAT, and GATv2 using PyTorch Geometric; GATv2 must support and save attention weights.
- The unified model has a projection, configurable GNN stack, anomaly head, and RCA head. Default hidden size 256, four heads, two layers, dropout 0.2; keep these in config.

## Required experiments and metrics

- Anomaly detection baselines: TF-IDF + Logistic Regression, TF-IDF + MLP, Qwen embedding + MLP, and Qwen + GATv2. Report precision, recall, F1, ROC-AUC, and PR-AUC when applicable; do not report accuracy alone.
- RCA ranking is case-level. Report Top-1/3/5, MRR, and Avg@1/3/5 when compatible with the official evaluator.
- Implement the requested TF-IDF/Qwen and MLP/GCN/GAT/GATv2 ablations, window-size, fault-type, cross-system, and representative-case analyses as resources allow.
- Do not fill result tables with placeholders presented as measurements. Poor results must be preserved honestly.

## Required project outputs

Keep the requested modular layout (`configs/`, `data/`, `models/`, `preprocessing/`, `experiments/`, `evaluation/`, `scripts/`, `results/`). Save configs, seed, dataset, split, model, metrics, timing, GPU memory, checkpoints, predictions, CSV/JSON/Markdown tables, figures, and TensorBoard or CSV training logs under `results/`.

## Execution gates

Follow phases in order: environment check; RE2-TT index download; schema/statistics; preprocessing; small (10-100 log) embedding validation; cached embeddings; graph construction; baselines; Qwen+MLP; GCN; GAT; GATv2; ablations and analyses; then additional systems. After each phase report what changed, disk/GPU usage, errors, and next step. Do not begin large-scale training until the user confirms the data format and statistics.

## Reproducible entry points

Maintain working entry points equivalent to:

```bash
python train.py --config configs/default.yaml
python evaluate.py --checkpoint <path>
python scripts/run_all_experiments.py
```

Document environment, data download, Qwen model, embeddings, graph construction, training, evaluation, baselines, ablations, and results in `README.md`.
