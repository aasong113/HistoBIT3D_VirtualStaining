#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run `eval_all_epochs_A2B_metrics_style.py` with convenient defaults.
#
# This behaves like `run_eval_all_epochs_A2B_metrics.sh` but adds optional style-fusion
# inference overrides:
#   - --style-fusion-state {auto,epoch,final,none}
#   - --style-fusion-inject {add,adain}
#   - --lambda-style-fusion FLOAT
#
# These overrides only apply when evaluating the style-fusion model
# `uvcgan2_3D_emb_sub_stylefusion.py` (config.model == "uvcgan2_3D_stylefusion").

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="/home/durrlab/Desktop/Anthony/UGVSM/3D_flow_consistent_UVCGANv2_vHE/scripts/eval_all_epochs_A2B_metrics_style.py"

# Keep these defaults minimal; override via CLI flags below.
DEFAULT_CHECKPOINTS_DIR="/home/durrlab/Desktop/Anthony/UGVSM/3D_flow_consistent_UVCGANv2_vHE/scripts/20260210_Inverted_MUSE_BIT2HE_crypts_lieberkuhn_train/outdir/20260211_Inverted_Combined_BIT2HE_crypts_lieberkuhn_Multiscale_Content_Train/20260211_duo_lieberkuhn_MUSEBIT_zspacing=2slices_lamsub=0p0_lamemb=0p0_lamMS=1p0_msC=16_lamSty=1p0/model_m(uvcgan2_3D_emb_sub_style_content)_d(basic)_g(vit-modnet)_uvcgan2-bn_(False:10.0:0.01:5e-05)/checkpoints"

CHECKPOINTS_DIR="${DEFAULT_CHECKPOINTS_DIR}"
TEST_A="/home/durrlab/Desktop/Anthony/data/duodenum_crypts_lieberkuhn_MUSE_BIT/BIT/trainA"
REAL_B="/home/durrlab/Desktop/Anthony/data/duodenum_crypts_lieberkuhn_MUSE_BIT/FFPE_HE/trainB"
OUTPUT_DIR="/home/durrlab/Desktop/Anthony/UGVSM/3D_flow_consistent_UVCGANv2_vHE/scripts/20260210_Inverted_MUSE_BIT2HE_crypts_lieberkuhn_train/outdir/20260211_Inverted_Combined_BIT2HE_crypts_lieberkuhn_Multiscale_Content_Train/20260211_duo_lieberkuhn_MUSEBIT_zspacing=2slices_lamsub=0p0_lamemb=0p0_lamMS=1p0_msC=16_lamSty=1p0/"
SPLIT="test"
BATCH_SIZE="1"
NUM_WORKERS="0"
N_EVAL=""
EPOCHS="50:80:5"
DATASET_NAME="cyclegan"
Z_SPACING="2"
SAMPLE_BASENAME=""
ALLOW_MISSING_METRICS="0"
ALLOW_UNPAIRED="1"
REQUIRE_PAIRED="0"
RESIZE_TO_REAL="0"
RESUME="0"
USE_AVG="0"
NO_KEEP_BEST="0"
SINGLE_GPU="1"

# Style-fusion specific overrides (only used by style models).
STYLE_FUSION_STATE="auto"      # auto|epoch|final|none
STYLE_FUSION_INJECT="adain"         # add|adain
LAMBDA_STYLE_FUSION=""         # float

usage() {
  cat <<'EOF'
Usage:
  run_eval_all_epochs_A2B_metrics_style.sh --checkpoints-dir PATH --test-a PATH --real-b PATH [options]

Required:
  --checkpoints-dir PATH  Path to model checkpoints/ (folder containing *_net_gen_ab.pth)
  --test-a PATH           Path to testA images OR CycleGAN root containing testA/
  --real-b PATH           Path to realB images (paired by basename)

Options:
  --output-dir PATH       Where to write outputs (default: <model_dir>/eval_all_epochs_metrics)
  --split {train,test,val} (default: test; ignored if --test-a points directly to trainA/testA/valA)
  --batch-size INT        (default: 1)
  --num-workers INT       DataLoader workers (default: 0)
  -n, --n-eval INT        Limit number of images translated (default: all)
  --epochs SPEC           "10,20,30" or "10:100:10"
  --dataset-name NAME     "cyclegan" or "adjacent-z-pairs" (default: cyclegan)
  --z-spacing INT         Only for adjacent-z-pairs (default: 1)
  --sample-basename NAME  Basename (no extension) of the sample image to save each epoch
  --allow-missing-metrics Write NaN for metrics whose deps are missing (lpips / torch-fidelity / clean-fid)
  --allow-unpaired        (default) If testA/realB are unpaired, write NaN for PSNR/SSIM/LPIPS and still compute FID/KID/IS
  --require-paired        Fail if no paired basenames exist between fake_b and real_b
  --use-avg               Use avg_gen_ab checkpoints/weights for inference (EMA)
  --no-keep-best          Do not keep fake_b images for best FID/KID/IS epochs
  --resize-to-real        If shapes differ, resize fake->real for PSNR/SSIM/LPIPS
  --resume                Skip epochs already present in metrics_by_epoch.txt
  --single-gpu            Disable DataParallel even if multiple GPUs are visible (use GPU 0 only)

Style-fusion overrides (only used for uvcgan2_3D_stylefusion):
  --style-fusion-state MODE     {auto,epoch,final,none} (default: auto)
  --style-fusion-inject MODE    {add,adain} (default: use checkpoint/config)
  --lambda-style-fusion FLOAT   Override lambda_style_fusion at inference time

Environment:
  CUDA_VISIBLE_DEVICES=0  Choose GPU (passed through to python)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoints-dir)
      [[ $# -ge 2 ]] || { echo "Error: --checkpoints-dir requires a PATH." >&2; usage; exit 2; }
      CHECKPOINTS_DIR="$2"; shift 2 ;;
    --test-a)
      [[ $# -ge 2 ]] || { echo "Error: --test-a requires a PATH." >&2; usage; exit 2; }
      TEST_A="$2"; shift 2 ;;
    --real-b)
      [[ $# -ge 2 ]] || { echo "Error: --real-b requires a PATH." >&2; usage; exit 2; }
      REAL_B="$2"; shift 2 ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "Error: --output-dir requires a PATH." >&2; usage; exit 2; }
      OUTPUT_DIR="$2"; shift 2 ;;
    --split)
      [[ $# -ge 2 ]] || { echo "Error: --split requires one of {train,test,val}." >&2; usage; exit 2; }
      SPLIT="$2"; shift 2 ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "Error: --batch-size requires an INT." >&2; usage; exit 2; }
      BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)
      [[ $# -ge 2 ]] || { echo "Error: --num-workers requires an INT." >&2; usage; exit 2; }
      NUM_WORKERS="$2"; shift 2 ;;
    -n|--n-eval)
      [[ $# -ge 2 ]] || { echo "Error: --n-eval requires an INT." >&2; usage; exit 2; }
      N_EVAL="$2"; shift 2 ;;
    --epochs)
      [[ $# -ge 2 ]] || { echo "Error: --epochs requires a SPEC." >&2; usage; exit 2; }
      EPOCHS="$2"; shift 2 ;;
    --dataset-name)
      [[ $# -ge 2 ]] || { echo "Error: --dataset-name requires 'cyclegan' or 'adjacent-z-pairs'." >&2; usage; exit 2; }
      DATASET_NAME="$2"; shift 2 ;;
    --z-spacing)
      [[ $# -ge 2 ]] || { echo "Error: --z-spacing requires an INT." >&2; usage; exit 2; }
      Z_SPACING="$2"; shift 2 ;;
    --sample-basename)
      [[ $# -ge 2 ]] || { echo "Error: --sample-basename requires a NAME." >&2; usage; exit 2; }
      SAMPLE_BASENAME="$2"; shift 2 ;;
    --allow-missing-metrics) ALLOW_MISSING_METRICS="1"; shift 1 ;;
    --allow-unpaired) ALLOW_UNPAIRED="1"; REQUIRE_PAIRED="0"; shift 1 ;;
    --require-paired) REQUIRE_PAIRED="1"; ALLOW_UNPAIRED="0"; shift 1 ;;
    --use-avg) USE_AVG="1"; shift 1 ;;
    --no-keep-best) NO_KEEP_BEST="1"; shift 1 ;;
    --resize-to-real) RESIZE_TO_REAL="1"; shift 1 ;;
    --resume) RESUME="1"; shift 1 ;;
    --single-gpu) SINGLE_GPU="1"; shift 1 ;;
    --style-fusion-state)
      [[ $# -ge 2 ]] || { echo "Error: --style-fusion-state requires a MODE." >&2; usage; exit 2; }
      STYLE_FUSION_STATE="$2"; shift 2 ;;
    --style-fusion-inject)
      [[ $# -ge 2 ]] || { echo "Error: --style-fusion-inject requires 'add' or 'adain'." >&2; usage; exit 2; }
      STYLE_FUSION_INJECT="$2"; shift 2 ;;
    --lambda-style-fusion)
      [[ $# -ge 2 ]] || { echo "Error: --lambda-style-fusion requires a FLOAT." >&2; usage; exit 2; }
      LAMBDA_STYLE_FUSION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${CHECKPOINTS_DIR}" || -z "${TEST_A}" || -z "${REAL_B}" ]]; then
  echo "Error: --checkpoints-dir, --test-a, and --real-b are required." >&2
  usage
  exit 2
fi

cmd=(python3 "${PY_SCRIPT}"
  --checkpoints-dir "${CHECKPOINTS_DIR}"
  --test-a "${TEST_A}"
  --real-b "${REAL_B}"
  --split "${SPLIT}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --dataset-name "${DATASET_NAME}"
  --z-spacing "${Z_SPACING}"
  --style-fusion-state "${STYLE_FUSION_STATE}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${N_EVAL}" ]]; then
  cmd+=(--n-eval "${N_EVAL}")
fi
if [[ -n "${EPOCHS}" ]]; then
  cmd+=(--epochs "${EPOCHS}")
fi
if [[ -n "${SAMPLE_BASENAME}" ]]; then
  cmd+=(--sample-basename "${SAMPLE_BASENAME}")
fi
if [[ "${ALLOW_MISSING_METRICS}" == "1" ]]; then
  cmd+=(--allow-missing-metrics)
fi
if [[ "${ALLOW_UNPAIRED}" == "1" ]]; then
  cmd+=(--allow-unpaired)
fi
if [[ "${REQUIRE_PAIRED}" == "1" ]]; then
  cmd+=(--require-paired)
fi
if [[ "${USE_AVG}" == "1" ]]; then
  cmd+=(--use-avg)
fi
if [[ "${NO_KEEP_BEST}" == "1" ]]; then
  cmd+=(--no-keep-best)
fi
if [[ "${RESIZE_TO_REAL}" == "1" ]]; then
  cmd+=(--resize-to-real)
fi
if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ "${SINGLE_GPU}" == "1" ]]; then
  cmd+=(--single-gpu)
fi

# Optional style-fusion behavior overrides.
if [[ -n "${STYLE_FUSION_INJECT}" ]]; then
  cmd+=(--style-fusion-inject "${STYLE_FUSION_INJECT}")
fi
if [[ -n "${LAMBDA_STYLE_FUSION}" ]]; then
  cmd+=(--lambda-style-fusion "${LAMBDA_STYLE_FUSION}")
fi

echo "[INFO] Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
