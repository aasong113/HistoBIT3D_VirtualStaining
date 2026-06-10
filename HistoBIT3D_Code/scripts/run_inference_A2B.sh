#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 /home/durrlab/Desktop/Anthony/UGVSM/3D_flow_consistent_UVCGANv2_vHE/scripts/translate_A2B.py \
  "/home/durrlab-asong/Anthony/3D_flow_consistent_UVCGANv2_vHE/outdir/20260113_Inverted_Combined_BIT2HE_normal_duodenum_only_crypts_Train_3DFlow/20260106_Inverted_Combined_BIT2HE_normal_duodenum_only_crypts_Train_3DFlow_zspacing=2slices_lambdsub=1p0_lambdemb=0p0/model_m(uvcgan2_3D_embedding_loss)_d(basic)_g(vit-modnet)_uvcgan2-bn_(False:10.0:0.01:5e-05)" \
  --split test \
  --epoch -1 \
  --data-root "/home/durrlab/Desktop/Anthony/data/20251225_duodenum_crypts/BIT/testA"

