# Project Handoff

Current phase: Phase 9 completed.

Dataset:
- Dataset: RE2-TT
- Cases requested: 90
- Cases with cached embeddings: 89
- Case without logs: re2tt_ts-auth-service_cpu_1
- Failed cases: 0

Embeddings:
- Path: data/processed/embeddings/qwen/60s
- Rows: 93590
- Dimension: 3584
- Pooling: mean
- Model revision: a09a35458c702b33eeacc393d103063234e8bc28
- Seed: 42

Phase 9 outputs:
- Metrics: `results/qwen_mlp/phase9_metrics.json`
- Markdown table: `results/qwen_mlp/phase9_metrics.md`
- Validation/test predictions: `results/qwen_mlp/phase9_predictions.csv`
- Run metadata: `results/qwen_mlp/phase9_run_metadata.json`
- Test metrics: precision 0.630208, recall 0.679775, F1 0.654054, ROC-AUC 0.668395, PR-AUC 0.724324.

Phases 10-16 outputs:
- Graph metrics and training history: `results/graph_experiments/phase10_12_metrics.json`
- Markdown comparison: `results/graph_experiments/phase10_16_summary.md`
- Anomaly predictions: `results/graph_experiments/phase10_12_test_predictions.csv`
- Case-level RCA rankings: `results/graph_experiments/phase10_12_rca_rankings.csv`
- GATv2 attention weights: `results/graph_experiments/phase12_gatv2_attention.npz`
- Fault-type metrics: `results/graph_experiments/phase14_fault_metrics.csv`
- Phase status: `results/graph_experiments/phase13_16_status.json`
- Checkpoints: `results/graph_experiments/phase10_gcn.pt`, `phase11_gat.pt`, `phase12_gatv2.pt`
- 2,139 graph snapshots were trained with seed 42. Cross-system Phase 15 remains unavailable because only RE2-TT has been prepared.

Next phase:
- Validate the graph training protocol and consider adding RE2-OB/RE2-SS before cross-system analysis.
