#!/usr/bin/env bash
set -euo pipefail

# Run 3D UVCGANv2 training using the `uvcgan2_3D_emb_sub_style_content.py` model
# (registered as `uvcgan2_3D_emb_sub_style_content` in `uvcgan2/cgan/__init__.py`).
#
# Usage:
#   ./run_train_3D_embedding_sub_stylecontent_TM.sh /abs/path/to/dataset_root
#
# Environment overrides (optional):
#   CUDA_VISIBLE_DEVICES=0           # GPU selection (defaults to 0)
#   WANDB_DIR=/tmp/wandb             # wandb log dir (defaults to /tmp/wandb)
#   BATCH_SIZE=1
#   Z_SPACING=2
#   LAMBDA_SUB_LOSS=0
#   LAMBDA_EMBEDDING_LOSS=0
#   LAMBDA_MULTISCALE_CONTENT=1.0
#   MULTISCALE_NUM_CHANNELS=8
#   MULTISCALE_SCALES=enc1,enc2,enc3,vit
#   LAMBDA_STYLE_LOSS=1.0
#   USE_STYLE_FUSION=1               # 0 disables, 1 enables (defaults to 1)
#   LAMBDA_STYLE_FUSION=1.0          # only meaningful if USE_STYLE_FUSION=1
#   STYLE_FUSION_INJECT=add          # 'add' or 'adain'

# Default dataset root for TM; you can still override by passing
# a different path as the first argument.
DEFAULT_ROOT_DATA_PATH="/home/durrlab/Desktop/Anthony/data/duodenum_crypts_lieberkuhn_MUSE_BIT_v2"

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
LAMBDA_MULTISCALE_CONTENT="${LAMBDA_MULTISCALE_CONTENT:-1.0}"
MULTISCALE_NUM_CHANNELS="${MULTISCALE_NUM_CHANNELS:-8}"
MULTISCALE_SCALES="${MULTISCALE_SCALES:-}"
LAMBDA_STYLE_LOSS="${LAMBDA_STYLE_LOSS:-1.0}"

USE_STYLE_FUSION="${USE_STYLE_FUSION:-1}"
LAMBDA_STYLE_FUSION="${LAMBDA_STYLE_FUSION:-1.0}"
STYLE_FUSION_INJECT="${STYLE_FUSION_INJECT:-adain}"

if [[ "${USE_STYLE_FUSION}" == "1" ]]; then
  STYLE_FUSION_FLAG="--use-style-fusion"
else
  STYLE_FUSION_FLAG="--no-style-fusion"
fi

# Resolve the training entrypoint relative to this script file so it works
# regardless of where you run it from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train_3D_embedding_style_content_TM.py"

cmd=(python3 "${TRAIN_PY}"
  --root_data_path "${ROOT_DATA_PATH}"
  --batch-size "${BATCH_SIZE}"
  --z-spacing "${Z_SPACING}"
  --lambda-sub-loss "${LAMBDA_SUB_LOSS}"
  --lambda-embedding-loss "${LAMBDA_EMBEDDING_LOSS}"
  --lambda-multiscale-content "${LAMBDA_MULTISCALE_CONTENT}"
  --multiscale-num-channels "${MULTISCALE_NUM_CHANNELS}"
  --lambda-style-loss "${LAMBDA_STYLE_LOSS}"
  ${STYLE_FUSION_FLAG}
  --lambda-style-fusion "${LAMBDA_STYLE_FUSION}"
  --style-fusion-inject "${STYLE_FUSION_INJECT}"
)

if [[ -n "${MULTISCALE_SCALES}" ]]; then
  cmd+=(--multiscale-scales "${MULTISCALE_SCALES}")
fi

printf '[INFO] Running:'
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
