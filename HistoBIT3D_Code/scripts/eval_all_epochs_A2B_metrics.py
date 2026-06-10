#!/usr/bin/env python3
"""
Evaluate a UVCGAN2(-3D) model across *all* checkpoint epochs by:
  1) Running A->B inference for every image in a provided testA folder
  2) Computing PSNR, SSIM, LPIPS, FID, KID, and Inception Score (IS)
     between generated fake_B images and provided real_B images
  3) Writing a per-epoch metrics table to a .txt file
  4) Plotting metrics vs epoch
  5) Deleting fake_B images after each epoch to save disk space,
     while keeping one consistent sample fake_B image per epoch.

This script is intentionally "junior engineer readable": it uses small
functions, straightforward control-flow, and in-line comments.

Example usage (CycleGAN-style data):
  python3 UGVSM/3D_flow_consistent_UVCGANv2_vHE/scripts/eval_all_epochs_A2B_metrics.py \\
    --checkpoints-dir "/path/to/model/checkpoints" \\
    --test-a "/path/to/cyclegan_root_or_testA" \\
    --real-b "/path/to/realB_images" \\
    --output-dir "/path/to/output_metrics_dir" \\
    --split test \\
    --batch-size 1

Notes:
  - For distribution metrics (FID/KID/IS), this script optionally uses:
      * torch_fidelity (preferred: computes FID/KID/IS in one call)
      * clean-fid (fallback: computes FID only)
    If these packages (and lpips) are not installed, run with
    --allow-missing-metrics to write NaN for missing metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tqdm
from PIL import Image

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from uvcgan2.consts import MERGE_NONE
from uvcgan2.config import Args
from uvcgan2.config.data_config import DatasetConfig
from uvcgan2.data import construct_data_loaders
from uvcgan2.eval.funcs import slice_data_loader, tensor_to_image
from uvcgan2.torch.funcs import get_torch_device_smart, seed_everything
from uvcgan2.cgan import construct_model


_KNOWN_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _strip_known_image_suffixes(name: str) -> str:
    """Repeatedly strip known image suffixes (handles ".tif.tif" etc.)."""
    base = os.path.basename(str(name))
    while True:
        root, ext = os.path.splitext(base)
        if ext.lower() in _KNOWN_IMAGE_EXTS and root:
            base = root
            continue
        return base


def _infer_cyclegan_root(path: str) -> str:
    """
    Accept either a CycleGAN root (containing testA/) or the testA/ directory
    itself, and always return the CycleGAN root.
    """
    path = os.path.abspath(os.path.expanduser(path))
    base = os.path.basename(path).lower()
    if base in {"testa", "traina", "vala", "testb", "trainb", "valb"}:
        return os.path.dirname(path)
    return path


def _split_to_cyclegan_dirname(split: str, domain_letter: str) -> str:
    # CycleGAN convention: e.g. "testA", "trainB"
    return f"{split.lower()}{domain_letter.upper()}"


def _resolve_cyclegan_root_and_split_for_test_a(test_a_path: str, split: str) -> Tuple[str, str]:
    """
    Resolve CycleGAN root + effective split for domain A.

    Behavior:
      - If `test_a_path` is a CycleGAN split dir like ".../trainA" or ".../testA",
        respect that folder and infer the split from its name (ignoring `split`).
      - Otherwise, treat `test_a_path` as either the CycleGAN root (containing trainA/testA/...)
        or a path under it, and use the provided `split`.
    """
    test_a_path = os.path.abspath(os.path.expanduser(test_a_path))
    base = os.path.basename(test_a_path).lower()
    inferred = {"traina": "train", "testa": "test", "vala": "val"}.get(base)
    if inferred is not None:
        return os.path.dirname(test_a_path), inferred
    return _infer_cyclegan_root(test_a_path), str(split)


def _list_image_files(path: str, recursive: bool) -> List[str]:
    """List image files (paths) under `path` with known extensions."""
    path = os.path.abspath(os.path.expanduser(path))
    result: List[str] = []

    if recursive:
        for root, _dirs, files in os.walk(path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in _KNOWN_IMAGE_EXTS:
                    result.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(path):
            if os.path.splitext(fname)[1].lower() in _KNOWN_IMAGE_EXTS:
                result.append(os.path.join(path, fname))

    return sorted(result)


def _dtype_to_float01(image: np.ndarray) -> np.ndarray:
    """Convert an image array to float32 in [0, 1]."""
    if image.dtype == np.uint8:
        return (image.astype(np.float32) / 255.0).clip(0.0, 1.0)
    if image.dtype == np.uint16:
        return (image.astype(np.float32) / 65535.0).clip(0.0, 1.0)
    if np.issubdtype(image.dtype, np.floating):
        # Assume already in [0,1] (common for generated PNGs) but clip defensively.
        return image.astype(np.float32).clip(0.0, 1.0)
    # Generic integer type: scale by max representable value.
    if np.issubdtype(image.dtype, np.integer):
        max_val = float(np.iinfo(image.dtype).max)
        return (image.astype(np.float32) / max_val).clip(0.0, 1.0)
    return image.astype(np.float32).clip(0.0, 1.0)


def _load_image_float_rgb(path: str) -> np.ndarray:
    """
    Load an image as float32 RGB in [0,1].
    - If the image is grayscale, it is expanded to 3 channels.
    - If the image has an alpha channel, alpha is discarded.
    """
    # PIL handles many formats (PNG/JPEG/TIF) and keeps behavior simple.
    img = Image.open(path)
    img = img.convert("RGB")
    arr = np.asarray(img)
    return _dtype_to_float01(arr)


def _save_float_rgb_as_png(image_float01: np.ndarray, out_path: str) -> None:
    """Save float [0,1] RGB image as uint8 PNG."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = (np.round(image_float01.clip(0.0, 1.0) * 255.0)).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def _try_import(name: str):
    """Import helper so we can give a clean error message."""
    try:
        return __import__(name)
    except Exception:  # pylint: disable=broad-except
        return None


@dataclass
class EpochMetrics:
    epoch: int
    num_pairs: int
    psnr: float
    ssim: float
    lpips: float
    fid: float
    kid: float
    inception_score: float


def _discover_epochs_from_checkpoints(checkpoints_dir: str, use_avg: bool) -> List[int]:
    """
    Find all epochs available in a checkpoints directory.

    The training code typically writes files like:
      0010_net_gen_ab.pth, 0010_net_avg_gen_ab.pth, ...
    We use either:
      - *_net_gen_ab.pth (default)
      - *_net_avg_gen_ab.pth (when --use-avg is set)
    """
    checkpoints_dir = os.path.abspath(os.path.expanduser(checkpoints_dir))
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"checkpoints dir not found: {checkpoints_dir}")

    if use_avg:
        epoch_re = re.compile(r"^(?P<epoch>\d+)_net_avg_gen_ab\.pth$")
    else:
        epoch_re = re.compile(r"^(?P<epoch>\d+)_net_gen_ab\.pth$")
    epochs: List[int] = []
    for fname in os.listdir(checkpoints_dir):
        m = epoch_re.match(fname)
        if m:
            epochs.append(int(m.group("epoch")))
    return sorted(set(epochs))


def _parse_epochs_arg(epochs_arg: Optional[str], available_epochs: Sequence[int]) -> List[int]:
    """
    Parse --epochs.
      - None: use all available
      - "10,20,30": explicit list
      - "10:100:10": python-like range start:stop:step (inclusive of stop if aligned)
    """
    if epochs_arg is None:
        return list(available_epochs)

    epochs_arg = epochs_arg.strip()
    if not epochs_arg:
        return list(available_epochs)

    if ":" in epochs_arg:
        parts = [p.strip() for p in epochs_arg.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid --epochs range: {epochs_arg}")
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("--epochs step cannot be 0")
        # Make stop inclusive when step divides evenly, which is the common expectation for epochs.
        seq = list(range(start, stop + (1 if step > 0 else -1), step))
        chosen = [e for e in seq if e in set(available_epochs)]
        return chosen

    chosen = []
    for token in epochs_arg.split(","):
        token = token.strip()
        if token:
            chosen.append(int(token))
    chosen = [e for e in chosen if e in set(available_epochs)]
    return sorted(set(chosen))


def _configure_testA_only_dataloader(
    args: Args,
    test_a_path: str,
    dataset_name: str,
    z_spacing: int,
    split: str,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    """
    Override the model's training data config so we can run inference on a user-provided testA folder.

    We preserve:
      - the training-time input shape
      - the training-time test transforms (normalization, resizing, etc.)
    """
    dataset_name = str(dataset_name)
    test_a_path = os.path.abspath(os.path.expanduser(test_a_path))

    # Preserve the shape + test transforms from the model's original domain A dataset.
    ref = args.config.data.datasets[0]
    shape = tuple(ref.shape) if isinstance(ref.shape, (list, tuple)) else ref.shape

    if dataset_name == "cyclegan":
        cyclegan_root, effective_split = _resolve_cyclegan_root_and_split_for_test_a(test_a_path, split)
        split = effective_split
        dataset_args = {"name": "cyclegan", "domain": "A", "path": cyclegan_root}
    elif dataset_name == "adjacent-z-pairs":
        dataset_args = {
            "name": "adjacent-z-pairs",
            "domain": "A",
            "path": test_a_path,
            "z_spacing": int(z_spacing),
        }
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # Replace datasets list with a single A-only dataset.
    args.config.data.datasets = [
        DatasetConfig(
            dataset=dataset_args,
            shape=shape,
            transform_train=None,
            transform_test=ref.transform_test,
        )
    ]
    args.config.data.merge_type = MERGE_NONE
    args.config.batch_size = int(batch_size)

    loader = construct_data_loaders(args.config.data, args.config.batch_size, split=split)
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    # Many datasets support "inference mode" to return filenames.
    if hasattr(loader.dataset, "set_inference"):
        loader.dataset.set_inference(True)

    # We need a collate_fn that preserves per-sample filenames.
    def inference_collate_fn(batch):
        first = batch[0]

        # CycleGAN/image folder datasets in inference mode:
        #   (tensor, filename)
        if isinstance(first, (list, tuple)) and len(first) == 2:
            images = [item[0] for item in batch]
            names = [item[1] for item in batch]
            return images, names

        # AdjacentZPairDataset returns dict:
        #   { "z1": tensor, "z2": tensor, "z1_name": str, "z2_name": str, ... }
        if isinstance(first, dict):
            images = []
            names = []
            for item in batch:
                if "z1" in item:
                    images.append(item["z1"])
                    names.append(item.get("z1_name", ""))
                if "z2" in item:
                    images.append(item["z2"])
                    names.append(item.get("z2_name", ""))
            return images, names

        raise TypeError(f"Unsupported batch item type: {type(first)}")

    # Wrap the dataset in a DataLoader with our custom collate_fn.
    # Default to num_workers=0 for portability (some environments disallow multiprocessing semaphores).
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=args.config.batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=inference_collate_fn,
    )


def _is_finite_number(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:  # pylint: disable=broad-except
        return False


def _metric_is_better(metric: str, value: float, best_value: Optional[float]) -> bool:
    if best_value is None:
        return True
    if metric in {"fid", "kid"}:
        return value < best_value
    if metric in {"is"}:
        return value > best_value
    raise ValueError(f"Unknown metric: {metric}")


def _safe_rmtree(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def _safe_move_dir(src_dir: str, dst_dir: str) -> None:
    """
    Move a directory, falling back to copy+delete if needed (e.g. across filesystems).
    """
    _safe_rmtree(dst_dir)
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    try:
        shutil.move(src_dir, dst_dir)
    except Exception:  # pylint: disable=broad-except
        shutil.copytree(src_dir, dst_dir)
        _safe_rmtree(src_dir)


def _update_best_pointer(best_root: str, metric: str, epoch: int) -> None:
    """
    Create/replace a symlink best_<metric> -> epoch_####.
    If symlinks aren't supported, write a small text file instead.
    """
    os.makedirs(best_root, exist_ok=True)
    target_dir = os.path.join(best_root, f"epoch_{epoch:04d}")
    link_path = os.path.join(best_root, f"best_{metric}")
    txt_path = os.path.join(best_root, f"best_{metric}.txt")

    # Remove previous link/file if present.
    for p in (link_path, txt_path):
        if os.path.islink(p) or os.path.isfile(p):
            with contextlib.suppress(Exception):
                os.remove(p)
        elif os.path.isdir(p):
            _safe_rmtree(p)

    try:
        rel = os.path.relpath(target_dir, best_root)
        os.symlink(rel, link_path)
    except Exception:  # pylint: disable=broad-except
        with open(txt_path, "wt", encoding="utf-8") as f:
            f.write(f"epoch: {epoch}\n")
            f.write(f"path: {target_dir}\n")


def _save_tensor_as_png(tensor_chw_float01: torch.Tensor, out_path: str) -> None:
    """Save a (C,H,W) tensor (float in [0,1]) as a PNG."""
    img = tensor_to_image(tensor_chw_float01)  # -> HxWxC float
    img = np.clip(img, 0.0, 1.0)
    img_u8 = np.round(255.0 * img).astype(np.uint8)
    Image.fromarray(img_u8).save(out_path)


def _translate_all_A_to_fakeB(
    model,
    loader: torch.utils.data.DataLoader,
    out_fake_b_dir: str,
    n_eval: Optional[int],
    ext: str,
) -> None:
    """
    Run inference on the full A dataset and save only fake_b images.
    """
    os.makedirs(out_fake_b_dir, exist_ok=True)

    data_it, steps = slice_data_loader(loader, batch_size=loader.batch_size, n_samples=n_eval)
    for images, names in tqdm.tqdm(data_it, total=steps, desc="A->B inference"):
        # Our collate_fn returns a list[Tensor] so we can keep names.
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        if isinstance(names, str):
            names = [names]

        # Move batch to GPU/CPU device once per batch.
        images = images.to(model.device, non_blocking=True)

        # Run the model forward pass without gradients.
        model.set_input(images, domain=0)  # domain=0 means "A"
        model.forward_nograd()

        fake_b = getattr(model.images, "fake_b", None)
        if fake_b is None:
            raise RuntimeError("Model did not produce 'fake_b' in model.images. Is this an A->B model?")

        # Save fake_b outputs using original filenames.
        for i, name in enumerate(names):
            base = _strip_known_image_suffixes(name)
            out_path = os.path.join(out_fake_b_dir, f"{base}.{ext}")
            _save_tensor_as_png(fake_b[i].detach().cpu(), out_path)


def _build_basename_to_path_map(image_paths: Sequence[str]) -> Dict[str, str]:
    """
    Build a mapping basename -> filepath.
    If duplicates exist, we keep the first (and warn later by counting).
    """
    mapping: Dict[str, str] = {}
    dup_count = 0
    for p in image_paths:
        base = _strip_known_image_suffixes(os.path.basename(p))
        if base in mapping:
            dup_count += 1
            continue
        mapping[base] = p

    if dup_count > 0:
        print(f"[WARN] Found {dup_count} duplicate basenames; kept the first occurrence.")
    return mapping


def _compute_pairwise_metrics(
    fake_b_dir: str,
    real_b_dir: str,
    allow_missing_metrics: bool,
    resize_to_real: bool,
    allow_unpaired: bool,
) -> Tuple[int, float, float, float]:
    """
    Compute PSNR, SSIM, and LPIPS on paired images matched by basename.
    Returns: (num_pairs, mean_psnr, mean_ssim, mean_lpips)
    """
    fake_paths = _list_image_files(fake_b_dir, recursive=False)
    real_paths = _list_image_files(real_b_dir, recursive=False)
    fake_map = _build_basename_to_path_map(fake_paths)
    real_map = _build_basename_to_path_map(real_paths)

    common = sorted(set(fake_map).intersection(real_map))
    if not common:
        if allow_unpaired:
            print(
                "[WARN] No paired images found between fake_b and real_b; "
                "paired metrics (PSNR/SSIM/LPIPS) will be NaN."
            )
            return 0, float("nan"), float("nan"), float("nan")
        raise RuntimeError(
            "No paired images found between fake_b and real_b. "
            "Ensure filenames match between testA and realB, or re-run with --allow-unpaired."
        )

    # Local imports so the script can still run when only plotting/resuming.
    have_skimage = True
    try:
        from skimage.metrics import peak_signal_noise_ratio as _psnr  # type: ignore
        from skimage.metrics import structural_similarity as _ssim  # type: ignore
    except Exception:  # pylint: disable=broad-except
        have_skimage = False
        if allow_missing_metrics:
            print("[WARN] scikit-image not installed; PSNR/SSIM will be NaN.")
        else:
            raise RuntimeError(
                "Missing dependency: scikit-image (skimage). Install it (e.g. `pip install scikit-image`) "
                "or re-run with --allow-missing-metrics."
            )

    # LPIPS is optional (it requires the external lpips package).
    lpips_mod = _try_import("lpips")
    lpips_model = None
    if lpips_mod is None:
        if allow_missing_metrics:
            print("[WARN] lpips not installed; LPIPS will be NaN.")
        else:
            raise RuntimeError(
                "Missing dependency: lpips. Install it (e.g. `pip install lpips`) "
                "or re-run with --allow-missing-metrics."
            )
    else:
        lpips_model = lpips_mod.LPIPS(net="alex")
        lpips_model.eval()

    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []

    for base in tqdm.tqdm(common, desc="Pair metrics (PSNR/SSIM/LPIPS)"):
        fake_img = _load_image_float_rgb(fake_map[base])
        real_img = _load_image_float_rgb(real_map[base])

        # These metrics require images to have the same shape.
        if fake_img.shape != real_img.shape:
            if not resize_to_real and (have_skimage or lpips_model is not None):
                raise RuntimeError(
                    f"Shape mismatch for '{base}': fake {fake_img.shape} vs real {real_img.shape}. "
                    "Re-run with --resize-to-real to resize fake->real for metrics."
                )

            if resize_to_real:
                # Resize fake -> real resolution (keeps code simple; may slightly change metrics).
                pil = Image.fromarray((fake_img * 255).astype(np.uint8))
                pil = pil.resize((real_img.shape[1], real_img.shape[0]), resample=Image.BICUBIC)
                fake_img = _dtype_to_float01(np.asarray(pil.convert("RGB")))

        if have_skimage:
            psnr_vals.append(float(_psnr(real_img, fake_img, data_range=1.0)))
            ssim_vals.append(float(_ssim(real_img, fake_img, channel_axis=-1, data_range=1.0)))
        else:
            psnr_vals.append(float("nan"))
            ssim_vals.append(float("nan"))

        if lpips_model is None:
            lpips_vals.append(float("nan"))
        else:
            # LPIPS expects tensors in [-1, 1], shape (N, 3, H, W).
            fake_t = torch.from_numpy(fake_img).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
            real_t = torch.from_numpy(real_img).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
            with torch.no_grad():
                lp = float(lpips_model(fake_t, real_t).item())
            lpips_vals.append(lp)

    return (
        len(common),
        float(np.nanmean(psnr_vals)),
        float(np.nanmean(ssim_vals)),
        float(np.nanmean(lpips_vals)),
    )


def _write_png_subset_dir(
    src_dir: str,
    basenames: Sequence[str],
    out_dir: str,
) -> None:
    """
    Create a directory of PNG files for a subset of images from `src_dir`.
    This is useful for FID/KID/IS libraries that expect typical RGB images.
    """
    os.makedirs(out_dir, exist_ok=True)
    src_paths = _list_image_files(src_dir, recursive=False)
    src_map = _build_basename_to_path_map(src_paths)

    for base in basenames:
        src_path = src_map.get(base)
        if src_path is None:
            continue
        img = _load_image_float_rgb(src_path)
        _save_float_rgb_as_png(img, os.path.join(out_dir, f"{base}.png"))


def _write_png_dir_from_paths(
    image_paths: Sequence[str],
    out_dir: str,
    limit: Optional[int],
) -> int:
    """
    Convert images to RGB PNGs with sequential names.
    Returns number of images written.
    """
    os.makedirs(out_dir, exist_ok=True)
    if limit is None:
        chosen = list(image_paths)
    else:
        chosen = list(image_paths)[: int(limit)]

    for idx, src_path in enumerate(chosen):
        img = _load_image_float_rgb(src_path)
        _save_float_rgb_as_png(img, os.path.join(out_dir, f"{idx:06d}.png"))
    return len(chosen)


def _compute_distribution_metrics(
    fake_b_dir: str,
    real_b_dir: str,
    allow_missing_metrics: bool,
) -> Tuple[float, float, float]:
    """
    Compute FID, KID, and Inception Score (IS).

    Preferred implementation: torch_fidelity (FID+KID+IS).
    Fallback: cleanfid (FID only).
    """
    # Local imports so the script can still run (e.g. plot-only) even if these
    # optional metric libraries are missing.
    try:
        from torch_fidelity import calculate_metrics as _tf_calculate_metrics  # type: ignore
    except Exception:  # pylint: disable=broad-except
        _tf_calculate_metrics = None

    try:
        from cleanfid import fid as _cleanfid_fid  # type: ignore
    except Exception:  # pylint: disable=broad-except
        _cleanfid_fid = None

    if _tf_calculate_metrics is None and _cleanfid_fid is None:
        if allow_missing_metrics:
            print("[WARN] torch_fidelity/cleanfid not installed; FID/KID/IS will be NaN.")
            return float("nan"), float("nan"), float("nan")
        raise RuntimeError(
            "Missing dependencies for FID/KID/IS: install torch-fidelity and/or clean-fid.\n"
            "  pip install torch-fidelity clean-fid\n"
            "Or re-run with --allow-missing-metrics."
        )

    fake_paths = _list_image_files(fake_b_dir, recursive=False)
    real_paths = _list_image_files(real_b_dir, recursive=False)
    if not fake_paths or not real_paths:
        raise RuntimeError("No images available to compute distribution metrics.")

    with tempfile.TemporaryDirectory(prefix="uvcgan_metrics_") as tmp_root:
        tmp_fake = os.path.join(tmp_root, "fake")
        tmp_real = os.path.join(tmp_root, "real")

        # Write RGB PNGs so external metric libraries can read them reliably.
        # Keep counts balanced for stability and to bound runtime.
        keep_n = min(len(fake_paths), len(real_paths))
        _write_png_dir_from_paths(fake_paths, tmp_fake, limit=keep_n)
        _write_png_dir_from_paths(real_paths, tmp_real, limit=keep_n)

        # torch_fidelity computes all three metrics in one call (if installed).
        if _tf_calculate_metrics is not None:
            # kid_subset_size must not exceed number of images.
            kid_subset_size = min(keep_n, 1000)
            metrics = _tf_calculate_metrics(
                # NOTE: torch-fidelity computes Inception Score (IS) for input1.
                # We want IS for the generated images, so we pass fake as input1.
                # (FID/KID are symmetric, so swapping inputs does not change them.)
                input1=tmp_fake,
                input2=tmp_real,
                cuda=torch.cuda.is_available(),
                isc=True,
                fid=True,
                kid=True,
                verbose=False,
                kid_subset_size=kid_subset_size,
            )
            fid_val = float(metrics.get("frechet_inception_distance", float("nan")))
            kid_val = float(metrics.get("kernel_inception_distance_mean", float("nan")))
            is_val = float(metrics.get("inception_score_mean", float("nan")))
            return fid_val, kid_val, is_val

        # Fallback: cleanfid for FID only.
        fid_val = float(_cleanfid_fid.compute_fid(tmp_fake, tmp_real))  # type: ignore[union-attr]
        return fid_val, float("nan"), float("nan")


def _read_existing_epochs_from_txt(path: str) -> set:
    """Parse an existing metrics .txt file to support --resume."""
    if not os.path.exists(path):
        return set()
    epochs = set()
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("epoch"):
                continue
            parts = re.split(r"\\s+", line)
            with contextlib.suppress(Exception):
                epochs.add(int(parts[0]))
    return epochs


def _append_metrics_row_txt(path: str, row: EpochMetrics) -> None:
    is_new = not os.path.exists(path)
    with open(path, "at", encoding="utf-8") as f:
        if is_new:
            f.write("epoch\tpairs\tpsnr\tssim\tlpips\tfid\tkid\tis\n")
        f.write(
            f"{row.epoch}\t{row.num_pairs}\t"
            f"{row.psnr:.6f}\t{row.ssim:.6f}\t{row.lpips:.6f}\t"
            f"{row.fid:.6f}\t{row.kid:.6f}\t{row.inception_score:.6f}\n"
        )


def _load_metrics_table_txt(path: str) -> List[EpochMetrics]:
    """Load the final metrics table to plot it."""
    rows: List[EpochMetrics] = []
    if not os.path.exists(path):
        return rows

    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("epoch"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append(
                EpochMetrics(
                    epoch=int(parts[0]),
                    num_pairs=int(parts[1]),
                    psnr=float(parts[2]),
                    ssim=float(parts[3]),
                    lpips=float(parts[4]),
                    fid=float(parts[5]),
                    kid=float(parts[6]),
                    inception_score=float(parts[7]),
                )
            )
    return rows


def _plot_metrics(rows: Sequence[EpochMetrics], out_path: str) -> None:
    """Plot metrics vs epoch and save as an image."""
    if not rows:
        print("[WARN] No rows to plot.")
        return

    # Local import keeps base dependencies minimal.
    import matplotlib.pyplot as plt

    epochs = [r.epoch for r in rows]
    series = {
        "PSNR (↑)": [r.psnr for r in rows],
        "SSIM (↑)": [r.ssim for r in rows],
        "LPIPS (↓)": [r.lpips for r in rows],
        "FID (↓)": [r.fid for r in rows],
        "KID (↓)": [r.kid for r in rows],
        "IS (↑)": [r.inception_score for r in rows],
    }

    n = len(series)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (title, vals) in zip(axes, series.items()):
        ax.plot(epochs, vals, marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    # Hide unused subplot(s) if any.
    for ax in axes[len(series) :]:
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A->B inference for all epochs and compute PSNR/SSIM/LPIPS/FID/KID/IS."
    )
    parser.add_argument(
        "--checkpoints-dir",
        required=True,
        help="Path to the model checkpoints directory (the folder containing *_net_gen_ab.pth).",
        type=str,
    )
    parser.add_argument(
        "--test-a",
        required=True,
        help="Path to testA images (or a CycleGAN root containing testA/).",
        type=str,
    )
    parser.add_argument(
        "--real-b",
        required=True,
        help="Path to realB images used for metric comparisons.",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write metrics/plots (default: <model_dir>/eval_all_epochs_metrics).",
        type=str,
    )
    parser.add_argument(
        "--epochs",
        default=None,
        help="Epoch selection: '10,20,30' or '10:100:10' (default: all epochs in checkpoints-dir).",
        type=str,
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test", "val"],
        help=(
            "Split name (used when --test-a is a CycleGAN root). "
            "If --test-a points directly to a split folder (e.g. .../trainA), split is inferred."
        ),
        type=str,
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        help="Batch size for inference.",
        type=int,
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        help="DataLoader workers (default: 0).",
        type=int,
    )
    parser.add_argument(
        "-n",
        "--n-eval",
        default=None,
        help="Number of images to translate (default: all).",
        type=int,
    )
    parser.add_argument(
        "--dataset-name",
        default="cyclegan",
        choices=["cyclegan", "adjacent-z-pairs"],
        help="Dataset type used to load testA.",
        type=str,
    )
    parser.add_argument(
        "--z-spacing",
        default=1,
        help="z spacing (only for adjacent-z-pairs).",
        type=int,
    )
    parser.add_argument(
        "--allow-missing-metrics",
        action="store_true",
        default=False,
        help="If set, missing metric dependencies produce NaN instead of failing.",
    )
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        default=False,
        help="Allow unpaired testA/realB (paired metrics become NaN; distribution metrics still run). Default: allowed.",
    )
    parser.add_argument(
        "--require-paired",
        action="store_true",
        default=False,
        help="Require paired testA/realB basenames (fail if no pairs found).",
    )
    parser.add_argument(
        "--resize-to-real",
        action="store_true",
        default=False,
        help="If set, resize fake->real when computing paired metrics if shapes differ.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="If set, skip epochs already present in the output .txt file.",
    )
    parser.add_argument(
        "--use-avg",
        action="store_true",
        default=False,
        help="Use avg_gen_ab checkpoints/weights for inference (EMA). Default: use non-avg gen_ab.",
    )
    parser.add_argument(
        "--no-keep-best",
        action="store_true",
        default=False,
        help="Disable keeping translated images for the best FID/KID/IS epochs.",
    )
    parser.add_argument(
        "--sample-basename",
        default=None,
        help="Basename (no extension) of the sample image to save each epoch. Default: first common basename.",
        type=str,
    )
    return parser.parse_args()


def main() -> None:
    cmd = parse_args()

    checkpoints_dir = os.path.abspath(os.path.expanduser(cmd.checkpoints_dir))
    model_dir = os.path.dirname(checkpoints_dir.rstrip(os.sep))

    available_epochs = _discover_epochs_from_checkpoints(checkpoints_dir, use_avg=cmd.use_avg)
    if not available_epochs:
        hint = "avg" if cmd.use_avg else "non-avg"
        extra = ""
        if not cmd.use_avg:
            extra = " (If you only have *_net_avg_gen_ab.pth files, re-run with --use-avg.)"
        raise RuntimeError(f"No {hint} epochs found in checkpoints dir: {checkpoints_dir}{extra}")

    epochs = _parse_epochs_arg(cmd.epochs, available_epochs)
    if not epochs:
        raise RuntimeError("No epochs selected after applying --epochs filter.")

    output_dir = cmd.output_dir
    if output_dir is None:
        output_dir = os.path.join(model_dir, "eval_all_epochs_metrics")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Primary outputs.
    metrics_txt = os.path.join(output_dir, "metrics_by_epoch.txt")
    plot_path = os.path.join(output_dir, "metrics_by_epoch.png")
    samples_dir = os.path.join(output_dir, "samples_fake_b")
    os.makedirs(samples_dir, exist_ok=True)

    keep_best = not bool(cmd.no_keep_best)
    best_root = os.path.join(output_dir, "best_epochs")
    if keep_best:
        os.makedirs(best_root, exist_ok=True)
        print(f"[INFO] Keeping translated fake_b images for best epochs under: {best_root}")

    # Optionally resume by skipping epochs already recorded.
    done_epochs = _read_existing_epochs_from_txt(metrics_txt) if cmd.resume else set()
    if done_epochs:
        epochs = [e for e in epochs if e not in done_epochs]
        print(f"[INFO] Resume enabled: {len(done_epochs)} epochs already done; {len(epochs)} remaining.")
    if not epochs:
        print("[INFO] Nothing to do (all requested epochs already evaluated).")
        _plot_metrics(_load_metrics_table_txt(metrics_txt), plot_path)
        return

    # Make sure the two image sources exist.
    test_a_path = os.path.abspath(os.path.expanduser(cmd.test_a))
    real_b_dir = os.path.abspath(os.path.expanduser(cmd.real_b))
    if not os.path.exists(test_a_path):
        raise FileNotFoundError(f"--test-a not found: {test_a_path}")
    if not os.path.isdir(real_b_dir):
        raise FileNotFoundError(f"--real-b not found: {real_b_dir}")

    # Resolve where testA images actually come from (and what split is used).
    effective_split = cmd.split
    if cmd.dataset_name == "cyclegan":
        root, effective_split = _resolve_cyclegan_root_and_split_for_test_a(test_a_path, cmd.split)
        if effective_split != cmd.split:
            print(
                f"[INFO] --test-a points to '{os.path.basename(test_a_path)}'; "
                f"overriding --split={cmd.split} -> {effective_split}"
            )
        test_a_images_dir = os.path.join(root, _split_to_cyclegan_dirname(effective_split, "A"))
    else:
        test_a_images_dir = test_a_path

    test_a_files = _list_image_files(test_a_images_dir, recursive=False)
    if not test_a_files:
        raise RuntimeError(f"No images found in inferred testA directory: {test_a_images_dir}")
    real_b_files = _list_image_files(real_b_dir, recursive=False)
    if not real_b_files:
        raise RuntimeError(f"No images found in realB directory: {real_b_dir}")

    test_a_bases = [_strip_known_image_suffixes(os.path.basename(p)) for p in test_a_files]
    sample_base = cmd.sample_basename or test_a_bases[0]
    if sample_base not in set(test_a_bases):
        raise RuntimeError(
            f"--sample-basename '{sample_base}' not found in testA filenames."
        )

    # Load model config once and reuse it; we only swap checkpoints by epoch.
    args = Args.load(model_dir)
    device = get_torch_device_smart()
    seed_everything(args.config.seed)

    # Construct model once; we will call model.load(epoch) for each epoch.
    model = construct_model(args.savedir, args.config, is_train=False, device=device)
    model.eval()
    if not cmd.use_avg and hasattr(model, "avg_momentum"):
        # Many UVCGAN2 variants use avg_gen_ab when avg_momentum is not None.
        # For this script we default to non-average inference unless --use-avg is set.
        model.avg_momentum = None

    # Create dataloader once (A-only). This keeps IO + transforms consistent across epochs.
    loader = _configure_testA_only_dataloader(
        args=args,
        test_a_path=test_a_path,
        dataset_name=cmd.dataset_name,
        z_spacing=cmd.z_spacing,
        split=effective_split,
        batch_size=cmd.batch_size,
        num_workers=cmd.num_workers,
    )

    # Save a small "run header" for traceability.
    run_info_path = os.path.join(output_dir, "run_info.txt")
    with open(run_info_path, "wt", encoding="utf-8") as f:
        f.write(f"timestamp: {_dt.datetime.now().isoformat()}\n")
        f.write(f"model_dir: {model_dir}\n")
        f.write(f"checkpoints_dir: {checkpoints_dir}\n")
        f.write(f"test_a: {test_a_path}\n")
        f.write(f"real_b: {real_b_dir}\n")
        f.write(f"dataset_name: {cmd.dataset_name}\n")
        f.write(f"split_requested: {cmd.split}\n")
        f.write(f"split_used: {effective_split}\n")
        f.write(f"batch_size: {cmd.batch_size}\n")
        f.write(f"n_eval: {cmd.n_eval}\n")
        f.write(f"sample_basename: {sample_base}\n")

    print(f"[INFO] Writing metrics to: {metrics_txt}")
    print(f"[INFO] Saving sample fake_b images to: {samples_dir}")
    print(f"[INFO] Sample basename (constant across epochs): {sample_base}")

    # Main evaluation loop: translate -> metrics -> save -> cleanup.
    best_epoch_for: Dict[str, Optional[int]] = {"fid": None, "kid": None, "is": None}
    best_value_for: Dict[str, Optional[float]] = {"fid": None, "kid": None, "is": None}
    best_metrics_by_epoch: Dict[int, set] = {}

    for idx, epoch in enumerate(epochs, start=1):
        print(f"\n[INFO] Evaluating epoch {epoch} ({idx}/{len(epochs)})")

        # Load checkpoint weights for the chosen epoch.
        model.load(epoch)
        model.eval()

        # Per-epoch scratch directory so we can delete generated images afterwards.
        epoch_work_dir = os.path.join(output_dir, f"_tmp_epoch_{epoch:04d}")
        fake_b_dir = os.path.join(epoch_work_dir, "fake_b")
        os.makedirs(epoch_work_dir, exist_ok=True)

        # 1) Inference: generate fake_b images.
        _translate_all_A_to_fakeB(
            model=model,
            loader=loader,
            out_fake_b_dir=fake_b_dir,
            n_eval=cmd.n_eval,
            ext="png",
        )

        # 2) Save a single consistent sample output for this epoch.
        sample_src = os.path.join(fake_b_dir, f"{sample_base}.png")
        if not os.path.exists(sample_src):
            raise RuntimeError(
                f"Sample output not found after inference: {sample_src}\n"
                "If your outputs are not PNG, change the script or set --sample-basename."
            )
        sample_dst = os.path.join(samples_dir, f"epoch_{epoch:04d}_{sample_base}.png")
        shutil.copyfile(sample_src, sample_dst)

        # 3) Compute metrics.
        allow_unpaired = cmd.allow_unpaired or (not cmd.require_paired)

        num_pairs, psnr_val, ssim_val, lpips_val = _compute_pairwise_metrics(
            fake_b_dir=fake_b_dir,
            real_b_dir=real_b_dir,
            allow_missing_metrics=cmd.allow_missing_metrics,
            resize_to_real=cmd.resize_to_real,
            allow_unpaired=allow_unpaired,
        )
        fid_val, kid_val, is_val = _compute_distribution_metrics(
            fake_b_dir=fake_b_dir,
            real_b_dir=real_b_dir,
            allow_missing_metrics=cmd.allow_missing_metrics,
        )

        row = EpochMetrics(
            epoch=epoch,
            num_pairs=num_pairs,
            psnr=psnr_val,
            ssim=ssim_val,
            lpips=lpips_val,
            fid=fid_val,
            kid=kid_val,
            inception_score=is_val,
        )
        _append_metrics_row_txt(metrics_txt, row)

        # 4) Optionally keep fake_b images for best epochs (FID/KID/IS).
        if keep_best:
            improved: List[Tuple[str, float]] = []
            for metric, val in (("fid", fid_val), ("kid", kid_val), ("is", is_val)):
                if not _is_finite_number(val):
                    continue
                if _metric_is_better(metric, float(val), best_value_for[metric]):
                    improved.append((metric, float(val)))

            if improved:
                # Ensure epoch dir exists by moving fake_b out of the tmp dir.
                epoch_kept_dir = os.path.join(best_root, f"epoch_{epoch:04d}")
                epoch_kept_fake_b = os.path.join(epoch_kept_dir, "fake_b")
                _safe_move_dir(fake_b_dir, epoch_kept_fake_b)

                # Update pointers and prune old best epochs no longer needed.
                for metric, val in improved:
                    old_epoch = best_epoch_for[metric]
                    if old_epoch is not None:
                        old_set = best_metrics_by_epoch.get(old_epoch, set())
                        old_set.discard(metric)
                        best_metrics_by_epoch[old_epoch] = old_set
                        if not old_set:
                            _safe_rmtree(os.path.join(best_root, f"epoch_{old_epoch:04d}"))

                    best_epoch_for[metric] = epoch
                    best_value_for[metric] = val
                    best_metrics_by_epoch.setdefault(epoch, set()).add(metric)
                    _update_best_pointer(best_root, metric, epoch)

        # 5) Cleanup: delete per-epoch scratch dir to save space.
        shutil.rmtree(epoch_work_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6) Plot all metrics after finishing.
    _plot_metrics(_load_metrics_table_txt(metrics_txt), plot_path)
    print(f"\n[INFO] Done. Plot saved to: {plot_path}")
    if keep_best:
        fid_best = best_epoch_for.get("fid")
        kid_best = best_epoch_for.get("kid")
        is_best = best_epoch_for.get("is")
        if fid_best is None and kid_best is None and is_best is None:
            print("[WARN] No finite FID/KID/IS found; nothing was kept in best_epochs.")
        else:
            print("[INFO] Best epochs kept:")
            if fid_best is not None:
                print(f"  FID best: epoch {fid_best} (lower is better)")
            if kid_best is not None:
                print(f"  KID best: epoch {kid_best} (lower is better)")
            if is_best is not None:
                print(f"  IS  best: epoch {is_best} (higher is better)")


if __name__ == "__main__":
    main()
