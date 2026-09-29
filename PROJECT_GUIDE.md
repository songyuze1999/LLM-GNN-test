# LLM-GNN-RCA 当前项目说明与复现指南

本文把当前工作目录当作一个可检查的实验档案来解释。读者不需要先了解论文：先看“实验在解决什么问题”，再看文件，最后按复现命令执行即可。

## 1. 先说结论：实验是否全部完成

没有全部完成。当前目录已经完成了一条可运行的 **RE2-TT、60 秒窗口** 主流程：

1. RCAEval `cases.parquet` 索引检查和统计。
2. RE2-TT 90 个 failure case 的本地 telemetry 预处理。
3. 60 秒窗口的日志、指标和 trace caller/callee 边构造。
4. 89 个有日志 case 的 Qwen2.5-7B-Instruct frozen hidden-state embedding 缓存。
5. case-level train/validation/test split。
6. TF-IDF baseline、Qwen embedding + MLP。
7. GCN、GAT、GATv2 anomaly/RCA 实验，以及 GATv2 attention 权重保存。

以下部分仍不是完整论文实验：

- **Phase 15 跨系统实验**没有执行。当前只准备了 RE2-TT，没有 RE2-OB 和 RE2-SS 的对应预处理数据。
- 30 秒和 120 秒窗口只存在极少数验证目录，没有全体 case 的窗口、embedding、graph 和训练结果，因此窗口 ablation 不能宣称完成。
- 当前没有 `train.py`、`evaluate.py`、`scripts/run_all_experiments.py` 这三个 AGENTS.md 要求的兼容入口；本项目使用 `scripts/reproduce_all.sh` 和分阶段脚本。
- 没有独立的官方 RCAEval evaluator 实现；当前 RCA 是使用 case index 中的 `root_cause_service` 计算 Top-1/3/5 和 MRR。
- 当前结果没有系统的多随机种子、超参数搜索、完整 metrics feature ablation、完整论文图表和 TensorBoard 训练日志。
- `data/processed/cases_schema_validation.json` 报告了原始全数据索引中两个 RE1 时间边界异常。它们不属于本次 RE2-TT 训练数据，但不能被忽略。

因此，现有结果适合说明“这条 RE2-TT 原型管线已经跑通”，不适合直接写成“论文所有实验已经完成”。

## 2. 论文到底在做什么

任务有两个层次：

- **异常检测**：一个时间窗口是否已经进入故障阶段。输出 anomaly score，并报告 precision、recall、F1、ROC-AUC、PR-AUC。
- **根因定位（RCA）**：在故障 case 中，哪个微服务最可能是根因。输出服务排名，并报告 Top-1、Top-3、Top-5、MRR。

技术路线是：

```text
原始 RCAEval telemetry
    -> 按 injection_time 对齐的时间窗口
    -> 日志文本 --Qwen hidden state--> 服务节点语义向量
    -> trace caller/callee -----------> 服务调用图边
    -> GCN / GAT / GATv2 -------------> anomaly head + RCA node head
```

这里的节点是服务，不是单条日志。Qwen 不生成答案作为特征，而是对日志文本的 frozen hidden state 做 mean pooling。图边只来自 trace 中可观测的 parent span 与 serviceName 关系。

## 3. 环境和配置

### 3.1 推荐环境

项目要求使用 Conda 环境 `llm-gnn`。当前验证过的关键版本：

- Python 3.12.14
- PyTorch 2.9.1+cu130
- PyTorch Geometric 2.8.0.post1
- NVIDIA RTX 5090，CUDA runtime 13.0
- 系统 `nvcc` 是 12.8；这和 PyTorch wheel 的 CUDA runtime 不是同一件事

依赖清单在 [`requirements.txt`](requirements.txt)。不要因为 `nvcc` 版本不同而重装驱动、PyTorch 或系统 CUDA。

### 3.2 唯一实验配置

[`configs/default.yaml`](configs/default.yaml) 是主要配置文件：

- `seed: 42`：Python、NumPy、PyTorch、CUDA 使用的固定种子。
- `data`：原始数据、处理数据、cases index 和数据集名称。
- `preprocessing.window_seconds`：默认 60，允许 30/60/120。
- `llm`：Qwen 本地路径、revision、pooling、最大 token 长度、dtype 和 batch size。
- `model`：hidden size 256、4 heads、2 layers、dropout 0.2。
- `baselines`：TF-IDF 和 MLP 参数。
- `graph_experiments`：图模型 epoch、学习率、batch size、RCA loss 权重和阈值。

实验代码应从这里读参数，不应在代码中另写一套实验参数。

## 4. 当前目录的文件说明

下面按目录覆盖当前目录下的每一类文件。90 个 case 会产生大量同构文件；例如每个 `case.npy`、`case.parquet`、`case.json` 都遵循同一个 schema，文件名只是 case ID 的不同。这样解释可以覆盖每一个重复实例，而不把 90 个名字抄成没有信息的清单。

### 4.1 根目录和规则文件

| 文件 | 作用 | 论文用途 |
|---|---|---|
| [`AGENTS.md`](AGENTS.md) | 项目硬性规则：数据不可伪造、case-level split、seed、模型路线、执行 gates 和输出要求。 | 实验可信性、可复现性和数据泄漏约束。 |
| [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) | 本文档，解释文件、实验和复现流程。 | 实验方法附录和复现说明。 |
| [`configs/default.yaml`](configs/default.yaml) | 全部主要实验参数。 | 方法章节的参数来源。 |
| [`requirements.txt`](requirements.txt) | Conda 环境需要的 Python 依赖和 PyG wheel 说明。 | 环境复现。 |
| [`scripts/reproduce_all.sh`](scripts/reproduce_all.sh) | 从索引检查到 GCN/GAT/GATv2 的一键顺序入口。 | 复现实验主入口。 |

### 4.2 `models/`：模型代码

| 文件 | 作用 | 论文用途 |
|---|---|---|
| [`models/__init__.py`](models/__init__.py) | 标记 `models` 为 Python package。 | 无直接实验结果。 |
| [`models/llm_encoder.py`](models/llm_encoder.py) | `FrozenQwenEncoder` 加载本地 Qwen、冻结参数、tokenize 日志、执行 mean/first/last pooling，输出 float32 embedding 和 encoder metadata。 | 论文中的 frozen LLM 语义编码器。 |
| [`models/graph_models.py`](models/graph_models.py) | `GraphAnomalyRCA` 统一实现 GCN、GAT、GATv2；projection 后接可配置 GNN stack、graph anomaly head 和 node RCA head；GATv2 可返回 attention。 | 论文中的图模型对比和联合 anomaly/RCA 模型。 |

`models/__pycache__/` 下的 `.pyc` 是 Python 自动生成的字节码缓存，不是论文数据，也不应作为源码修改对象。

### 4.3 `preprocessing/`：窗口构造代码

| 文件 | 作用 | 论文用途 |
|---|---|---|
| [`preprocessing/__init__.py`](preprocessing/__init__.py) | 标记 preprocessing package。 | 无直接结果。 |
| [`preprocessing/window_builder.py`](preprocessing/window_builder.py) | 把 timestamp 转成 window ID；按 injection_time 标记 `normal`、`mixed`、`faulty`；从日志生成服务级计数和消息，从指标生成 long-form aggregate，从 trace parent span 推导 caller/callee 边。 | 数据预处理、标签定义和可靠服务依赖图的来源。 |

关键定义：窗口不是重新采样 telemetry；`window_start/end` 直接由原始时间戳和窗口宽度计算。`anomaly_label = 1` 表示窗口结束时间已经到达 injection time。

### 4.4 `scripts/`：可执行实验脚本

| 文件 | 作用 | 产生/读取的主要文件 |
|---|---|---|
| [`scripts/validate_cases.py`](scripts/validate_cases.py) | 读取 `data/raw/cases.parquet`，检查列、dtype、空值、重复 case、负计数、可用性和 injection 边界。 | 产生 `data/processed/cases_schema_validation.json`。 |
| [`scripts/analyze_cases.py`](scripts/analyze_cases.py) | 不读取 telemetry，只统计 RCAEval index 的数据集、系统、故障、根因、日志/trace/metric 数量。 | 产生 `data/processed/dataset_statistics.json`。 |
| [`scripts/preprocess_case.py`](scripts/preprocess_case.py) | 对一个指定 case 执行窗口化，支持 30/60/120 秒。适合调试或补建单 case。 | 产生该 case 的 `metric_windows.parquet`、`log_windows.parquet`、`window_index.parquet`、`trace_edges.parquet`。 |
| [`scripts/preprocess_re2tt.py`](scripts/preprocess_re2tt.py) | 对本地已有的全部 RE2-TT case 顺序执行窗口化；失败 case 写入 manifest，不删除原始数据。 | 产生 `data/processed/windows/` 和 `re2tt_preprocessing_manifest.json`。 |
| [`scripts/validate_qwen_embeddings.py`](scripts/validate_qwen_embeddings.py) | 从一个日志窗口选 10–100 条 normal/faulty 日志，验证 Qwen hidden state 能运行并保存 metadata。 | 产生 `data/processed/embeddings/validation/`。 |
| [`scripts/cache_qwen_embeddings.py`](scripts/cache_qwen_embeddings.py) | 逐 case 读取日志文本，调用 frozen Qwen，保存 embedding、无消息的行索引和 encoder metadata；已完成 case 默认跳过。 | 产生 `data/processed/embeddings/qwen/60s/`。 |
| [`scripts/build_service_graphs.py`](scripts/build_service_graphs.py) | 固定 seed 42，将相同 root-cause/fault/system group 放在同一 split；把 trace edges 和日志计数组装成服务节点和边表。当前明确只构建 60 秒。 | 产生 split parquet/json、每 case 的 nodes/edges、graph manifest。 |
| [`scripts/run_baselines.py`](scripts/run_baselines.py) | 在 case split 上训练 TF-IDF + Logistic Regression、TF-IDF + MLP、Qwen embedding + MLP。 | 产生 `results/baselines/phase8_*`。 |
| [`scripts/run_qwen_mlp.py`](scripts/run_qwen_mlp.py) | 独立重跑 Phase 9 的 Qwen mean-pooled embedding + standardized MLP，保存 train/validation/test 指标和预测。 | 产生 `results/qwen_mlp/phase9_*`。 |
| [`scripts/run_graph_experiments.py`](scripts/run_graph_experiments.py) | 读取 graph 和 Qwen embedding，按 case/window 构造 PyG Data，训练 GCN/GAT/GATv2，保存 anomaly、RCA、fault-type、checkpoint 和 GATv2 attention。 | 产生 `results/graph_experiments/`。 |
| [`scripts/reproduce_all.sh`](scripts/reproduce_all.sh) | 按正确依赖顺序调用上面 9 个阶段。 | 一键重现当前 RE2-TT 主流程。 |

`scripts/__pycache__/` 下的 `.pyc` 同样只是解释器缓存。

### 4.5 `data/raw/`：原始数据和不可修改的输入

| 文件模式 | 作用 | 论文用途 |
|---|---|---|
| [`data/raw/cases.parquet`](data/raw/cases.parquet) | RCAEval 官方 cases index。包含 `case`、`dataset`、`system_name`、`root_cause_service`、`fault`、`inject_time`、时间范围和 telemetry 数量。 | 数据集定义、case 标签、split 和统计的唯一来源。 |
| `data/raw/<case>/inject_time.txt` | 该 failure case 的注入时间。 | 正常/混合/故障窗口边界。 |
| `data/raw/<case>/logs.parquet` | 原始日志，至少含 timestamp、container_name、message。`re2tt_ts-auth-service_cpu_1` 没有该文件。 | Qwen 输入和日志计数特征。 |
| `data/raw/<case>/metrics.parquet` | 原始时间序列指标。 | 指标窗口数据；当前图模型默认没有把完整 metrics 向量接入。 |
| `data/raw/<case>/traces.parquet` | 原始 span trace，含 parent span、service、时间和状态码。 | 可靠 caller/callee 服务依赖边。 |

当前 raw 有 90 个 RE2-TT case 的 telemetry 目录，约 1.9 GB；另有全量 RCAEval index。原始数据不能覆盖、清洗后回写或删除。

### 4.6 `data/processed/`：中间数据

#### `data/processed/cases_schema_validation.json`

`validate_cases.py` 的机器可读检查结果。它记录 735 行、22 列的官方 index、各数据集计数、RE2-TT 故障/根因分布和时间边界异常。它不是训练标签的副本，也不是模型预测。

#### `data/processed/dataset_statistics.json`

`analyze_cases.py` 的统计结果。当前 RE2-TT 是 90 case、6 类故障各 15 个、5 个根因服务各 18 个；89 case 有日志、90 case 有 trace。

#### `data/processed/re2tt_preprocessing_manifest.json`

记录窗口预处理请求数、成功数、缺失日志 case 和失败 case。当前 90 个 case 完成，1 个 case 缺少日志，失败数为 0。

#### `data/processed/windows/<case>/<width>s/`

每个 parquet 的作用如下：

| 文件 | 内容 | 后续用途 |
|---|---|---|
| `window_index.parquet` | 每个时间窗口的事件数、相对 injection 时间、phase 和 anomaly label。 | 窗口索引和标签检查。 |
| `log_windows.parquet` | 每个 case/window/service 的日志消息、log/error/warning count、phase、label、fault 和根因字段。 | Qwen embedding、TF-IDF、节点计数特征。 |
| `metric_windows.parquet` | 每个 window/service/metric 的 mean/min/max/non-null 聚合。 | 后续可选 metrics ablation；当前图模型未完整使用。 |
| `trace_edges.parquet` | 每个 window caller/callee 的 call_count/error_count 以及相对时间和标签。 | graph edges 的直接输入。 |

当前完整主流程是 60 秒；目录中的少量 `30s`/`120s` 是单 case 试验，不代表完整窗口 ablation。

#### `data/processed/splits/`

| 文件 | 作用 |
|---|---|
| `re2tt_case_split.parquet` | 每个 case 的 `system`、`root_cause_service`、`fault`、`repetition`、`split_group` 和 `split`。窗口不会跨 split。 |
| `re2tt_case_split.json` | split 的摘要：30 个 group，train/validation/test case 数为 63/12/15。 |

#### `data/processed/embeddings/`

`validation/` 保存 10–100 条日志的小样本验证 embedding 和 metadata。

`qwen/60s/` 中每个有日志的 case 有三件配套文件：

- `<case>.npy`：float32，形状约为 `行数 x 3584` 的 Qwen mean-pooled hidden state。
- `<case>.parquet`：与 `.npy` 行号一一对应的 metadata，包含 case/window/service/phase/label，但不含原始 message。
- `<case>.json`：该 case 的 source path、source SHA256、embedding shape、encoder metadata、配置和 seed。

`manifest.json` 汇总了 90 个请求、89 个完成、1 个缺失日志 case，并记录 Qwen revision `a09a35458c702b33eeacc393d103063234e8bc28`、BF16、CUDA、mean pooling 和 hidden size 3584。

#### `data/processed/graphs/re2tt/60s/`

- `<case>/nodes.parquet`：每个 window 的服务节点，包含 `service`、`node_id`、窗口时间、phase、anomaly label、log/error/warning count。
- `<case>/edges.parquet`：每个 window 的 `src`/`dst`、caller/callee、call_count、error_count 和标签。
- `manifest.json`：graph schema、源 trace SHA256、每个 case 的窗口/节点/边数量、失败 case 和 split metadata。

当前共生成 90 个 case graph；节点/边是可复现的 trace 结果，不是人工补的服务依赖。

### 4.7 `results/`：论文实验结果

#### `results/baselines/`

- `phase8_metrics.json`：TF-IDF Logistic Regression、TF-IDF MLP、Qwen MLP 的 validation/test 指标。
- `phase8_test_predictions.csv`：每个 test window 的模型 score 和真实 label。
- `phase8_run_metadata.json`：seed、split、pooling 和缺失 case 说明。

#### `results/qwen_mlp/`

- `phase9_metrics.json`：独立 Qwen + MLP 的 train/validation/test 指标、窗口数量、GPU 和 embedding manifest。
- `phase9_metrics.md`：适合直接检查的指标表。
- `phase9_predictions.csv`：validation/test 每窗口预测。
- `phase9_run_metadata.json`：运行元数据。

#### `results/graph_experiments/`

- `phase10_gcn.pt`、`phase11_gat.pt`、`phase12_gatv2.pt`：三个模型的 PyTorch state dict checkpoint。
- `phase10_12_metrics.json`：训练历史、anomaly 指标、RCA Top-k/MRR 和配置。
- `phase10_12_test_predictions.csv`：三个模型的 test window score。
- `phase10_12_rca_rankings.csv`：每个 test case 的根因排名。
- `phase12_gatv2_attention.npz`：GATv2 attention 权重数组。
- `phase14_fault_metrics.csv`：按 fault 和模型分组的 test 指标。
- `phase13_16_status.json`：Phase 13–16 的执行状态和限制。
- `phase10_16_summary.md`：模型对比表，并明确 Phase 15 不可用。

这些结果是真实运行产物。较差的 ROC-AUC 或 RCA 不能被替换成占位符或“理论值”。

### 4.8 `data/models/Qwen2.5-7B-Instruct/`

这是本地 Qwen2.5-7B-Instruct 模型快照，不是本项目训练出的 checkpoint：

- `model-00001-of-00004.safetensors` 到 `model-00004-of-00004.safetensors`：四个模型权重 shard。
- `model.safetensors.index.json`：参数名到 shard 的索引。
- `config.json`：模型结构配置。
- `tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`：tokenizer 文件。
- `generation_config.json`：生成默认配置；本实验主要取 hidden state，不依赖生成答案。
- `README.md`、`LICENSE`、`.gitattributes`：上游模型说明、许可和 Git LFS 配置。
- `.cache/huggingface/`：Hugging Face 下载缓存和 revision metadata，不参与模型逻辑。

### 4.9 `migration/`

这些文件记录环境迁移和交接，不是训练输入：

- [`migration/NEW_SERVER_SETUP.md`](migration/NEW_SERVER_SETUP.md)：新服务器环境设置说明。
- [`migration/python-version.txt`](migration/python-version.txt)：Python 版本记录。
- [`migration/nvidia-smi.txt`](migration/nvidia-smi.txt)：GPU/驱动记录。
- [`migration/llm-gnn.pip-freeze.txt`](migration/llm-gnn.pip-freeze.txt)：环境包冻结列表。
- [`migration/llm-gnn.environment.yml`](migration/llm-gnn.environment.yml)：Conda 环境描述。
- [`migration/HANDOFF.md`](migration/HANDOFF.md)：当前阶段、数据摘要、产物和下一步。

## 5. 如何复现

### 5.1 复现前检查

先进入仓库并激活环境：

```bash
cd /workspace/lunwen
conda activate llm-gnn
python -c "import torch, torch_geometric, pandas; print(torch.__version__, torch.cuda.is_available(), torch_geometric.__version__)"
```

应看到 PyTorch 2.9.1+cu130、PyG 2.8.0.post1，并且 GPU 可用。CPU 可以运行 sklearn 和小规模检查，但 Qwen embedding 和图训练会更慢。

### 5.2 一键复现当前已完成主流程

```bash
cd /workspace/lunwen
conda activate llm-gnn
./scripts/reproduce_all.sh
```

脚本按以下顺序执行：

```text
validate_cases
 -> analyze_cases
 -> preprocess_re2tt
 -> validate_qwen_embeddings
 -> cache_qwen_embeddings
 -> build_service_graphs
 -> run_baselines
 -> run_qwen_mlp
 -> run_graph_experiments
```

脚本默认使用 `configs/default.yaml`，默认不会重复计算已经缓存的 Qwen case。可以指定配置：

```bash
CONFIG=configs/default.yaml ./scripts/reproduce_all.sh
```

如果 Python 不在 PATH：

```bash
PYTHON_BIN=/path/to/conda/envs/llm-gnn/bin/python ./scripts/reproduce_all.sh
```

### 5.3 分阶段复现命令

只检查索引：

```bash
python scripts/validate_cases.py
python scripts/analyze_cases.py
```

处理全部本地 RE2-TT telemetry：

```bash
python scripts/preprocess_re2tt.py --config configs/default.yaml
```

处理一个 case 或另一个窗口：

```bash
python scripts/preprocess_case.py --config configs/default.yaml \
  --case re2tt_ts-auth-service_cpu_2 --window-seconds 30
```

验证小样本 embedding：

```bash
python scripts/validate_qwen_embeddings.py --config configs/default.yaml \
  --sample-size 32
```

缓存 Qwen embedding：

```bash
python scripts/cache_qwen_embeddings.py --config configs/default.yaml
```

如果某个 case 的缓存损坏或配置发生改变，才使用 `--force`；不要为了普通重跑删除 raw、processed、checkpoint 或 results。

构建 split 和服务图：

```bash
python scripts/build_service_graphs.py --config configs/default.yaml
```

运行 baseline、Phase 9 和 Phase 10–16：

```bash
python scripts/run_baselines.py --config configs/default.yaml
python scripts/run_qwen_mlp.py --config configs/default.yaml
python scripts/run_graph_experiments.py --config configs/default.yaml
```

### 5.4 从空目录开始时缺少什么

仓库内已经保存当前服务器上的 raw telemetry 和 Qwen 权重，但代码没有实现 RCAEval/Hugging Face 的下载脚本。若迁移到另一台机器，需要先按官方 RCAEval/Hugging Face 数据格式准备：

1. `data/raw/cases.parquet`。
2. `data/raw/<case>/inject_time.txt`。
3. 对目标 case 可用的 `metrics.parquet`、`logs.parquet`、`traces.parquet`。
4. `data/models/Qwen2.5-7B-Instruct/` 的完整本地模型。

数据格式不一致时，应先检查列名和 schema，不能猜测字段或伪造缺失 trace/label。

## 6. 怎样理解一条结果

以 Qwen + MLP 为例：

1. `log_windows.parquet` 把同一 window 内同一 service 的多条日志合并成文本。
2. Qwen 对文本输出 3584 维 hidden-state mean pooling。
3. 同一 window 的多个 service embedding 再 mean pooling 成窗口向量。
4. StandardScaler + MLP 输出 anomaly score。
5. threshold 0.5 产生 precision/recall/F1；连续 score 产生 ROC-AUC 和 PR-AUC。

以图模型为例：

1. 一个 graph snapshot 对应一个 case 的一个 window。
2. 节点特征是 Qwen service embedding，加上 `log1p(log_count)`、error ratio、warning ratio。
3. edge_index 来自 trace caller/callee。
4. graph mean pooling 进入 anomaly head；节点进入 RCA head。
5. test case 内按服务聚合 node RCA score，再计算真实 root cause 的 rank。

注意：窗口级 anomaly 指标和 case 级 RCA 指标不是同一件事，不能用一个替代另一个。

## 7. 当前结果的安全解读

- Phase 8/9 的 Qwen MLP test F1 为 0.654054，ROC-AUC 为 0.668395，PR-AUC 为 0.724324。
- 当前 graph 对比的结果在 [`results/graph_experiments/phase10_16_summary.md`](results/graph_experiments/phase10_16_summary.md)。
- GAT/GATv2 在当前训练设置下出现高 recall、低区分度倾向；这是真实结果，不代表模型已经达到论文目标。
- RCA Top-k 是当前实现计算的 case-level 排名，不等同于已通过官方 RCAEval evaluator。
- 只有 RE2-TT 的结果不能支持“跨系统泛化”结论。

## 8. 完成论文实验还需要什么

要把“原型已跑通”推进到“论文实验完整”，至少还要：

1. 补齐 RE2-OB、RE2-SS 的索引筛选、telemetry、窗口、embedding 和 graph。
2. 对 30/60/120 秒做完整、同一 split 规则下的 window ablation。
3. 实现并验证官方 RCA evaluator 兼容的排名输入和指标。
4. 增加 TF-IDF/Qwen、MLP/GCN/GAT/GATv2、metrics/no-metrics 等正式 ablation 脚本。
5. 增加 checkpoint 加载评估入口 `evaluate.py`、统一 `train.py` 和 `run_all_experiments.py`。
6. 保存每次运行的完整 config、GPU memory、timing、checkpoint、prediction、CSV/JSON/Markdown 表和训练日志。
7. 对现有 graph node label、RCA loss 和 test 聚合策略做独立验证，确认没有把 case label 或 root cause 信息意外泄漏进输入。

因此，当前代码可以让读者完整理解并复现已完成的 RE2-TT 主实验，但不能诚实地声称整篇论文的所有必要实验已经结束。
