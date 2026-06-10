#!/usr/bin/env bash
set -euo pipefail

# Run 3D UVCGANv2 training using the `uvcgan2_3D_emb_sub_stylefusion.py` model
# (registered as `uvcgan2_3D_stylefusion` in `uvcgan2/cgan/__init__.py`).
#
# Usage:
#   ./run_train_3D_embedding_sub_stylefusion_TM.sh /abs/path/to/dataset_root
#
# Environment overrides (optional):
#   CUDA_VISIBLE_DEVICES=0           # GPU selection (defaults to 0)
#   WANDB_DIR=/tmp/wandb             # wandb log dir (defaults to /tmp/wandb)
#   BATCH_SIZE=1
#   Z_SPACING=2
#   LAMBDA_SUB_LOSS=0
#   LAMBDA_EMBEDDING_LOSS=0
#   LAMBDA_STYLE_LOSS=1.0
#   USE_STYLE_FUSION=0               # 0 disables, 1 enables
#   LAMBDA_STYLE_FUSION=0.0          # only meaningful if USE_STYLE_FUSION=1
#   STYLE_FUSION_INJECT=adain        # 'add' or 'adain'

# Default dataset root for this environment; you can still override by passing
# a different path as the first argument.
DEFAULT_ROOT_DATA_PATH="/home/durrlab-asong/Anthony/duodenum_submucosa_crypts_full_MUSE_BIT"

ROOT_DATA_PATH="${1:-${DEFAULT_ROOT_DATA_PATH}}"
if [[ -z "${ROOT_DATA_PATH}" ]]; then
  echo "Usage: $(basename "$0") [/abs/path/to/dataset_root]" >&2
  exit 2
fi

# Defaults (can be overridden via env vars above).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"

BATCH_SIZE="${BATCH_SIZE:-1}"
Z_SPACING="${Z_SPACING:-2}"
LAMBDA_SUB_LOSS="${LAMBDA_SUB_LOSS:-0}"
LAMBDA_EMBEDDING_LOSS="${LAMBDA_EMBEDDING_LOSS:-0}"
LAMBDA_STYLE_LOSS="${LAMBDA_STYLE_LOSS:-1.0}"
STYLE_FUSION_INJECT="${STYLE_FUSION_INJECT:-adain}"

# Resolve the training entrypoint relative to this script file so it works
# regardless of where you run it from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train_3D_embedding_style_loss_WSL.py"

python3 "${TRAIN_PY}" \
  --root_data_path "${ROOT_DATA_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --z-spacing "${Z_SPACING}" \
  --lambda-sub-loss "${LAMBDA_SUB_LOSS}" \
  --lambda-embedding-loss "${LAMBDA_EMBEDDING_LOSS}" \
  --lambda-style-loss "${LAMBDA_STYLE_LOSS}" \
  --style-fusion-inject "${STYLE_FUSION_INJECT}"
