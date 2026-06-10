#!/usr/bin/env bash
set -euo pipefail

# Wrapper for pretrain_BIT2HE.py with user-configurable dataset paths,
# checkpoint cadence, output directory name, and run description.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$/home/durrlab-asong/Anthony/3D_flow_consistent_UVCGANv2_vHE/scripts/pretrain_BIT2HE.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ROOT_DATA_PATH="/home/durrlab-asong/Anthony/duodenum_crypts_lieberkuhn_MUSE_BIT_v2"
DOMAIN_A_PATH="BIT"
DOMAIN_B_PATH="FFPE_HE"
OUTDIR_NAME="pretrain_test"
CHECKPOINT_EVERY="5"
DESCRIPTION=""
GEN="uvcgan2"
BATCH_SIZE="2"
OUTDIR_ROOT=""

usage() {
  cat <<'EOF2'
Usage:
  run_pretrain_BIT2HE.sh [options]

Data options (choose one mode):
  --root-data-path PATH    Root that contains default subfolders:
                           kidney_normal_BIT-invBIT_BIT and kidney_normal_FFPE_HE
  --domain-a-path PATH     Explicit domain A path (CycleGAN-style dataset root)
  --domain-b-path PATH     Explicit domain B path (CycleGAN-style dataset root)

Run options:
  --checkpoint-every INT   Save checkpoint every N epochs (default: 5)
  --outdir-name NAME       Outdir name under UVCGAN2_OUTDIR/ROOT_OUTDIR
  --description TEXT       Label/description stored with the run
  --outdir-root PATH       Optional ROOT outdir override (exports UVCGAN2_OUTDIR)
  --gen NAME               Generator preset (default: uvcgan2)
  --batch-size INT         Batch size (default: 32)
  --python BIN             Python executable (default: python3)
  -h, --help               Show this help and exit
EOF2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-data-path) ROOT_DATA_PATH="$2"; shift 2 ;;
    --domain-a-path) DOMAIN_A_PATH="$2"; shift 2 ;;
    --domain-b-path) DOMAIN_B_PATH="$2"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    --outdir-name) OUTDIR_NAME="$2"; shift 2 ;;
    --description) DESCRIPTION="$2"; shift 2 ;;
    --outdir-root) OUTDIR_ROOT="$2"; shift 2 ;;
    --gen) GEN="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${ROOT_DATA_PATH}" && ( -z "${DOMAIN_A_PATH}" || -z "${DOMAIN_B_PATH}" ) ]]; then
  echo "Error: provide --root-data-path OR both --domain-a-path and --domain-b-path." >&2
  usage
  exit 2
fi

if [[ -n "${OUTDIR_ROOT}" ]]; then
  export UVCGAN2_OUTDIR="${OUTDIR_ROOT}"
fi

cmd=(
  "${PYTHON_BIN}" "${PY_SCRIPT}"
  --gen "${GEN}"
  --batch-size "${BATCH_SIZE}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --outdir-name "${OUTDIR_NAME}"
)

if [[ -n "${DESCRIPTION}" ]]; then
  cmd+=(--description "${DESCRIPTION}")
fi

if [[ -n "${ROOT_DATA_PATH}" ]]; then
  cmd+=(--root_data_path "${ROOT_DATA_PATH}")
fi

if [[ -n "${DOMAIN_A_PATH}" ]]; then
  cmd+=(--domain-a-path "${DOMAIN_A_PATH}")
fi

if [[ -n "${DOMAIN_B_PATH}" ]]; then
  cmd+=(--domain-b-path "${DOMAIN_B_PATH}")
fi

printf '[INFO] Running:'
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
