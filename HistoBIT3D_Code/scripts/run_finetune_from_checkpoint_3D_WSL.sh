#!/usr/bin/env bash
set -euo pipefail

# Fine-tune a 3D UVCGANv2 variant from an existing checkpoint epoch on a new dataset.
#
# Required inputs:
#   1) Base checkpoints dir (the folder containing e.g. 0010_net_gen_ab.pth)
#   2) New dataset location, either:
#        a) A single root path (legacy): expects BIT/trainA and FFPE_HE under it
#        b) Explicit split paths: exact trainA path and exact trainB path
#
# Usage:
#   bash run_finetune_from_checkpoint_3D_WSL.sh \
#     --base-checkpoints-dir /abs/path/to/base_run/checkpoints \
#     --root-data-path /abs/path/to/new_dataset_root \  # (legacy root layout)
#     --base-epoch 10 \
#     --model uvcgan2_3D_stylefusion
#
# Or (explicit split paths):
#   bash run_finetune_from_checkpoint_3D_WSL.sh \
#     --base-checkpoints-dir /abs/path/to/base_run/checkpoints \
#     --trainA-path /abs/path/to/BIT/trainA \
#     --trainB-path /abs/path/to/FFPE_HE/trainB \
#     --base-epoch 10 \
#     --model uvcgan2_3D_stylefusion
#
# Notes:
#   - This is fine-tuning in a new output directory (optimizers are re-initialized).
#   - For stylefusion runs, the script also loads style_fusion_state for the chosen epoch.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/finetune_from_checkpoint_3D.py"

# Defaults (override via flags).
BASE_CHECKPOINTS_DIR="/home/durrlab-asong/Anthony/3D_flow_consistent_UVCGANv2_vHE/outdir/20260201_Inverted_Combined_BIT2HE_normal_duodenum_only_crypts_Train_3DFlow/20260201_duodenum_only_crypts_3DFlow_zspacing=2slices_lamsub=0p0_lamemb=0p0_lamSty=1p0/model_m(uvcgan2_3D_stylefusion)_d(basic)_g(vit-modnet)_uvcgan2-bn_(False:10.0:0.01:5e-05)/checkpoints"
BASE_EPOCH="50"                # int or "last"
ROOT_DATA_PATH=""
TRAINA_PATH="/home/durrlab-asong/Anthony/duodenum_crypts_full_data/Paired_MUSE_BIT/trainA"
TRAINB_PATH="/home/durrlab-asong/Anthony/duodenum_crypts_full_data/FFPE_HE/trainB"
MODEL="uvcgan2_3D_stylefusion"                     # auto|uvcgan2_3D_stylefusion|uvcgan2_3D_embedding_loss|uvcgan2_3D_subtraction_loss

BATCH_SIZE="1"
EPOCHS="50"
CHECKPOINT_EVERY="1"
STEPS_PER_EPOCH="10"
NUM_WORKERS="1"

Z_SPACING="2"
LAMBDA_CYCLE="10.0"
LAMBDA_GP="0.01"
LR_GEN="5e-5"
LR_DISC="1e-4"

LAMBDA_SUB_LOSS="0.0"
LAMBDA_EMBEDDING_LOSS="0.0"

LAMBDA_STYLE_FUSION="0.0"
STYLE_FUSION_INJECT="adain"
LAMBDA_STYLE_LOSS="1.0"

INIT_FROM_AVG="0"
ALLOW_PARTIAL="0"

OUTDIR="/home/durrlab-asong/Anthony/3D_flow_consistent_UVCGANv2_vHE/outdir/20260201_Inverted_Combined_BIT2HE_normal_duodenum_only_crypts_Train_3DFlow"
RUN_NAME="20260201_duodenum_crypts_transfer_learn_with_MUSE_BIT_paired_data"

usage() {
  cat <<'EOF'
Usage:
  run_finetune_from_checkpoint_3D_WSL.sh --base-checkpoints-dir PATH (--root-data-path PATH | --trainA-path PATH --trainB-path PATH) [options]

Required:
  --base-checkpoints-dir PATH   Base run checkpoints/ dir (contains 00XX_net_gen_ab.pth)
  Dataset location (choose one):
    --root-data-path PATH       New dataset root (expects BIT/trainA and FFPE_HE under it)
    --trainA-path PATH          Exact path to domain A trainA directory
    --trainB-path PATH          Exact path to domain B trainB directory (or its CycleGAN root containing trainB/)

Options:
  --base-epoch EPOCH            int or "last" (default: last)
  --model NAME                  auto|uvcgan2_3D_stylefusion|uvcgan2_3D_embedding_loss|uvcgan2_3D_subtraction_loss
  --batch-size INT              (default: 1)
  --epochs INT                  (default: 200)
  --checkpoint-every INT        (default: 10)
  --steps-per-epoch INT         (default: 2000)
  --num-workers INT             (default: 1)
  --z-spacing INT               (default: 2)
  --lambda-cycle FLOAT          (default: 10.0)
  --lambda-gp FLOAT             (default: 0.01)
  --lr-gen FLOAT                (default: 5e-5)
  --lr-disc FLOAT               (default: 1e-4)
  --lambda-sub-loss FLOAT       (default: 0.0)
  --lambda-embedding-loss FLOAT (default: 0.0)
  --lambda-style-fusion FLOAT   (stylefusion only; default: 0.0)
  --style-fusion-inject MODE    add|adain (stylefusion only; default: adain)
  --lambda-style-loss FLOAT     (stylefusion only; default: 1.0)
  --init-from-avg               Initialize gen weights from avg_gen_* (EMA) if available
  --allow-partial               strict=False when loading checkpoint state_dicts
  --outdir PATH                 Output root (default: uses UVCGAN2_OUTDIR / "outdir")
  --run-name NAME               Custom run folder name under outdir
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-checkpoints-dir) BASE_CHECKPOINTS_DIR="$2"; shift 2 ;;
    --base-epoch) BASE_EPOCH="$2"; shift 2 ;;
    --root-data-path) ROOT_DATA_PATH="$2"; shift 2 ;;
    --trainA-path) TRAINA_PATH="$2"; shift 2 ;;
    --trainB-path) TRAINB_PATH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    --steps-per-epoch) STEPS_PER_EPOCH="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --z-spacing) Z_SPACING="$2"; shift 2 ;;
    --lambda-cycle) LAMBDA_CYCLE="$2"; shift 2 ;;
    --lambda-gp) LAMBDA_GP="$2"; shift 2 ;;
    --lr-gen) LR_GEN="$2"; shift 2 ;;
    --lr-disc) LR_DISC="$2"; shift 2 ;;
    --lambda-sub-loss) LAMBDA_SUB_LOSS="$2"; shift 2 ;;
    --lambda-embedding-loss) LAMBDA_EMBEDDING_LOSS="$2"; shift 2 ;;
    --lambda-style-fusion) LAMBDA_STYLE_FUSION="$2"; shift 2 ;;
    --style-fusion-inject) STYLE_FUSION_INJECT="$2"; shift 2 ;;
    --lambda-style-loss) LAMBDA_STYLE_LOSS="$2"; shift 2 ;;
    --init-from-avg) INIT_FROM_AVG="1"; shift 1 ;;
    --allow-partial) ALLOW_PARTIAL="1"; shift 1 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${BASE_CHECKPOINTS_DIR}" ]]; then
  echo "Error: --base-checkpoints-dir is required." >&2
  usage
  exit 2
fi

# Dataset location validation:
# - If explicit split paths are provided, require both.
# - Otherwise require the legacy root path.
if [[ -n "${TRAINA_PATH}" || -n "${TRAINB_PATH}" ]]; then
  if [[ -z "${TRAINA_PATH}" || -z "${TRAINB_PATH}" ]]; then
    echo "Error: If using explicit paths, both --trainA-path and --trainB-path are required." >&2
    usage
    exit 2
  fi
else
  if [[ -z "${ROOT_DATA_PATH}" ]]; then
    echo "Error: Either --root-data-path OR (--trainA-path and --trainB-path) must be provided." >&2
    usage
    exit 2
  fi
fi

cmd=(python3 "${PY_SCRIPT}"
  --base-checkpoints-dir "${BASE_CHECKPOINTS_DIR}"
  --base-epoch "${BASE_EPOCH}"
  --model "${MODEL}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --steps-per-epoch "${STEPS_PER_EPOCH}"
  --num-workers "${NUM_WORKERS}"
  --z-spacing "${Z_SPACING}"
  --lambda-cycle "${LAMBDA_CYCLE}"
  --lambda-gp "${LAMBDA_GP}"
  --lr-gen "${LR_GEN}"
  --lr-disc "${LR_DISC}"
  --lambda-sub-loss "${LAMBDA_SUB_LOSS}"
  --lambda-embedding-loss "${LAMBDA_EMBEDDING_LOSS}"
  --lambda-style-fusion "${LAMBDA_STYLE_FUSION}"
  --style-fusion-inject "${STYLE_FUSION_INJECT}"
  --lambda-style-loss "${LAMBDA_STYLE_LOSS}"
)

# Pass dataset paths.
# If explicit paths are set, use them and ignore --root-data-path.
if [[ -n "${TRAINA_PATH}" && -n "${TRAINB_PATH}" ]]; then
  cmd+=(--trainA-path "${TRAINA_PATH}" --trainB-path "${TRAINB_PATH}")
else
  cmd+=(--root-data-path "${ROOT_DATA_PATH}")
fi

if [[ "${INIT_FROM_AVG}" == "1" ]]; then
  cmd+=(--init-from-avg)
fi
if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
  cmd+=(--allow-partial)
fi
if [[ -n "${OUTDIR}" ]]; then
  cmd+=(--outdir "${OUTDIR}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  cmd+=(--run-name "${RUN_NAME}")
fi

echo "[INFO] Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
