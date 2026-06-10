#!/usr/bin/env bash
set -euo pipefail

# Reconstruct BIT and vHE stacks from patch folders and metadata.
#
# Usage:
#   ./run_reconstruct_bit_vhe.sh /abs/path/to/BIT_folder /abs/path/to/vHE_folder
#
# Optional env overrides:
#   METADATA_GLOB="*_patches_stitch_metadata.txt"
#   BIT_OUT_NAME="BIT_reconstructed"
#   VHE_OUT_NAME="vHE_reconstructed"
#   BIT_EXT="tif"
#   VHE_EXT="png"

BIT_FOLDER="/home/durrlab-asong/Anthony/duodenum_crypts_full_data/BIT/trainA"
VHE_FOLDER="/home/durrlab-asong/Anthony/3D_flow_consistent_UVCGANv2_vHE/outdir/20260201_Inverted_Combined_BIT2HE_normal_duodenum_only_crypts_Train_3DFlow/20260201_duodenum_only_crypts_3DFlow_zspacing=2slices_lamsub=0p0_lamemb=0p0_lamSty=1p0/best_epochs/epoch_0050/fake_b"

if [[ -z "${BIT_FOLDER}" || -z "${VHE_FOLDER}" ]]; then
  echo "Usage: $(basename "$0") /abs/path/to/BIT_folder /abs/path/to/vHE_folder" >&2
  exit 2
fi

METADATA_GLOB="${METADATA_GLOB:-*_patches_stitch_metadata.txt}"
BIT_OUT_NAME="${BIT_OUT_NAME:-BIT_reconstructed}"
VHE_OUT_NAME="${VHE_OUT_NAME:-vHE_reconstructed}"
BIT_EXT="${BIT_EXT:-tif}"
VHE_EXT="${VHE_EXT:-png}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/reconstruct_bit_vhe_from_metadata.py"

python3 "${PY_SCRIPT}" \
  --bit-folder "${BIT_FOLDER}" \
  --vhe-folder "${VHE_FOLDER}" \
  --metadata-glob "${METADATA_GLOB}" \
  --bit-out-name "${BIT_OUT_NAME}" \
  --vhe-out-name "${VHE_OUT_NAME}" \
  --bit-ext "${BIT_EXT}" \
  --vhe-ext "${VHE_EXT}"
