# New Server Setup

This document restores the project on a new GPU server and resumes from
Phase 7. The archive contains the project code, `AGENTS.md`, RCAEval data,
processed windows, cached Qwen embeddings, and the local Qwen model.

## 1. Prepare the server

Use a working directory with at least 25 GB free space, preferably more for
future checkpoints and results.

```bash
mkdir -p /workspace/transfer
cd /workspace/transfer
```

Install `rclone` if it is not already installed:

```bash
curl https://rclone.org/install.sh | sudo bash
```

Configure a Google Drive remote named `gdrive`:

```bash
rclone config
```

Choose Google Drive, then authorize the same Google account that owns the
`codex-lunwen/phase6-20260927` folder. A headless server can use the browser
authorization flow on another machine and paste the returned token.

## 2. Download the migration files

```bash
rclone copy \
  gdrive:codex-lunwen/phase6-20260927 \
  /workspace/transfer \
  --progress \
  --transfers 1
```

Expected files:

```text
lunwen_phase6_20260927.tar.zst
lunwen_phase6_20260927.tar.zst.sha256
lunwen_phase6_migration_20260927.tar.zst
lunwen_phase6_migration_20260927.tar.zst.sha256
```

## 3. Verify and extract

```bash
cd /workspace/transfer
sha256sum -c lunwen_phase6_20260927.tar.zst.sha256
sha256sum -c lunwen_phase6_migration_20260927.tar.zst.sha256

cd /workspace
tar -I zstd -xf /workspace/transfer/lunwen_phase6_20260927.tar.zst
tar -I zstd -xf /workspace/transfer/lunwen_phase6_migration_20260927.tar.zst
```

Check the key files:

```bash
test -f /workspace/lunwen/AGENTS.md
test -f /workspace/lunwen/migration/HANDOFF.md
test -f /workspace/lunwen/data/processed/embeddings/qwen/60s/manifest.json
test -d /workspace/lunwen/data/models/Qwen2.5-7B-Instruct
```

## 4. Recreate the Python environment

Do not copy the old Conda environment directory. Recreate it on the new
server and keep the pinned CUDA/PyTorch versions from the project rules.

```bash
conda create -n llm-gnn python=3.12 -y
conda activate llm-gnn
cd /workspace/lunwen
python -m pip install -r requirements.txt
```

If the server uses the same CUDA 13.0 PyTorch wheel setup, install the PyG
extension wheels listed in `requirements.txt` as needed.

Verify the environment and GPU:

```bash
conda run --no-capture-output -n llm-gnn python - <<'PY'
import torch
import transformers
import pandas
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("pandas:", pandas.__version__)
PY
nvidia-smi
```

## 5. Resume with Codex

Set the working directory to `/workspace/lunwen`, then ask Codex to read:

```text
Please read AGENTS.md and migration/HANDOFF.md.
Phase 6 is complete. Continue with Phase 7: construct service graphs from trace edges.
Do not begin large-scale training before checking the graph schema and case-level splits.
```

The cached embeddings are under
`data/processed/embeddings/qwen/60s/`. The cache contains 89 cases and 93,590
windows. `re2tt_ts-auth-service_cpu_1` has no logs and is recorded in the
embedding manifest.
