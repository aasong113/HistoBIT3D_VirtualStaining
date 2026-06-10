#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run `eval_all_epochs_A2B_metrics.py` with convenient defaults.
# You typically only need to provide:
#   - --test-a : your testA folder (or CycleGAN root containing testA/)
#   - --real-b : the matching realB folder
#
# Example:
#   bash UGVSM/3D_flow_consistent_UVCGANv2_vHE/scripts/run_eval_all_epochs_A2B_metrics.sh \
#     --test-a "/home/durrlab/Desktop/Anthony/data/XYZ/testA" \
#     --real-b "/home/durrlab/Desktop/Anthony/data/XYZ/testB" \
#     --split test \
#     --batch-size 1 \
#     --resume

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="/home/durrlab/Desktop/Anthony/UGVSM/3D_flow_consistent_UVCGANv2_vHE/scripts/eval_all_epochs_A2B_metrics.py"

# Default to the checkpoints path you referenced in your prompt; override with --checkpoints-dir.
DEFAULT_CHECKPOINTS_DIR="/home/durrlab/Desktop/Anthony/UGVSM/UVCGANv2_vHE/outdir/20251225_Inverted_combined_BIT2HE_duodenum_crypts/20251225_Inverted_combined_BIT2HE_duodenum_crypts_train/model_m(uvcgan2)_d(basic)_g(vit-modnet)_uvcgan2-bn_False-10.0-0.01-5e-05/checkpoints"

CHECKPOINTS_DIR="${DEFAULT_CHECKPOINTS_DIR}"
TEST_A="/home/durrlab/Desktop/Anthony/data/20251225_duodenum_crypts/BIT/trainA"
REAL_B="/home/durrlab/Desktop/Anthony/data/20251225_duodenum_crypts/FFPE_HE/trainB"
OUTPUT_DIR="/home/durrlab/Desktop/Anthony/UGVSM/UVCGANv2_vHE/outdir/20251225_Inverted_combined_BIT2HE_duodenum_crypts/20251225_Inverted_combined_BIT2HE_duodenum_crypts_train"
SPLIT="test"
BATCH_SIZE="1"
NUM_WORKERS="0"
N_EVAL=""
EPOCHS="10:200:10"
DATASET_NAME="cyclegan"
Z_SPACING="1"
SAMPLE_BASENAME=""
ALLOW_MISSING_METRICS="0"
ALLOW_UNPAIRED="1"
REQUIRE_PAIRED="0"
RESIZE_TO_REAL="0"
RESUME="0"
USE_AVG="0"
NO_KEEP_BEST="0"

usage() {
  cat <<'EOF'
Usage:
  run_eval_all_epochs_A2B_metrics.sh --test-a PATH --real-b PATH [options]

Required:
  --test-a PATH           Path to testA images OR CycleGAN root containing testA/
  --real-b PATH           Path to realB images (paired by basename)

Options:
  --checkpoints-dir PATH  Path to model checkpoints/ (default is hardcoded in this script)
  --output-dir PATH       Where to write outputs (default: <model_dir>/eval_all_epochs_metrics)
  --split {train,test,val} (default: test; ignored if --test-a points directly to trainA/testA/valA)
  --batch-size INT        (default: 1)
  --num-workers INT       DataLoader workers (default: 0)
  -n, --n-eval INT        Limit number of images translated (default: all)
  --epochs SPEC           "10,20,30" or "10:100:10" (default: all available)
  --dataset-name NAME     "cyclegan" or "adjacent-z-pairs" (default: cyclegan)
  --z-spacing INT         Only for adjacent-z-pairs (default: 1)
  --sample-basename NAME  Basename (no extension) of the sample image to save each epoch
  --allow-missing-metrics Write NaN for metrics whose deps are missing (lpips / torch-fidelity / clean-fid)
  --allow-unpaired        (default) If testA/realB are unpaired, write NaN for PSNR/SSIM/LPIPS and still compute FID/KID/IS
  --require-paired        Fail if no paired basenames exist between fake_b and real_b
  --use-avg               Use avg_gen_ab checkpoints/weights for inference (EMA). Default: non-avg gen_ab.
  --no-keep-best          Do not keep fake_b images for best FID/KID/IS epochs
  --resize-to-real        If shapes differ, resize fake->real for PSNR/SSIM/LPIPS
  --resume                Skip epochs already present in metrics_by_epoch.txt

Environment:
  CUDA_VISIBLE_DEVICES=0  Choose GPU (passed through to python)
EOF
}

if [[ $# -eq 0 ]]; then
  if [[ -z "${TEST_A}" || -z "${REAL_B}" ]]; then
    usage
    exit 1
  fi
  echo "[INFO] No CLI args provided; using defaults embedded in the script."
fi

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
      [[ $# -ge 2 ]] || { echo "Error: --epochs requires a SPEC (e.g. '10,20,30' or '10:100:10')." >&2; usage; exit 2; }
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
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${TEST_A}" || -z "${REAL_B}" ]]; then
  echo "Error: --test-a and --real-b are required." >&2
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

echo "[INFO] Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
