#!/usr/bin/env python3
"""
Fine-tune a UVCGAN2 3D variant from an existing checkpoint epoch on a new dataset.

Supports these models (config.model names):
  - uvcgan2_3D_stylefusion        (uvcgan2/cgan/uvcgan2_3D_emb_sub_stylefusion.py)
  - uvcgan2_3D_embedding_loss     (uvcgan2/cgan/uvcgan2_3D_embedding_loss.py)
  - uvcgan2_3D_subtraction_loss   (uvcgan2/cgan/uvcgan2_3D_subtraction_loss.py)

This script:
  1) Loads the base run config from <base_model_dir>/config.json
  2) Overrides dataset paths + a few training hyperparameters (CLI)
  3) Constructs a fresh model in a new output directory
  4) Loads network weights from the specified base checkpoint epoch
  5) (Stylefusion only) loads the matching style_fusion_state for that epoch if available
  6) Starts training on the new dataset

Notes:
  - This is "fine-tune", not "resume": optimizers/schedulers are NOT loaded.
  - Default dataset layout matches your existing training scripts:
      root_data_path/
        BIT/trainA
        FFPE_HE
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from itertools import islice
from typing import Any, Dict, Optional

import tqdm
import torch

# Ensure we import the in-repo uvcgan2 package (mirrors your other scripts).
_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_FILE, "..", ".."))
import sys  # isort: skip

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
print("Using uvcgan2 from:", _REPO_ROOT)

from uvcgan2.consts import ROOT_OUTDIR  # noqa: E402
from uvcgan2.config import Args  # noqa: E402
from uvcgan2.config.data_config import DatasetConfig  # noqa: E402
from uvcgan2.data import construct_data_loaders  # noqa: E402
from uvcgan2.cgan import construct_model  # noqa: E402
from uvcgan2.torch.funcs import get_torch_device_smart, seed_everything  # noqa: E402
from uvcgan2.train.callbacks import TrainingHistory  # noqa: E402
from uvcgan2.train.metrics import LossMetrics  # noqa: E402


_KNOWN_MODELS = {
    "uvcgan2_3D_stylefusion",
    "uvcgan2_3D_embedding_loss",
    "uvcgan2_3D_subtraction_loss",
}


def _discover_last_epoch(checkpoints_dir: str, prefer_avg: bool) -> int:
    """
    Find the last epoch by scanning for *_net_gen_ab.pth or *_net_avg_gen_ab.pth.
    """
    checkpoints_dir = os.path.abspath(os.path.expanduser(checkpoints_dir))
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"checkpoints dir not found: {checkpoints_dir}")

    if prefer_avg:
        r = re.compile(r"^(?P<epoch>\\d+)_net_avg_gen_ab\\.pth$")
    else:
        r = re.compile(r"^(?P<epoch>\\d+)_net_gen_ab\\.pth$")

    epochs = []
    for fname in os.listdir(checkpoints_dir):
        m = r.match(fname)
        if m:
            epochs.append(int(m.group("epoch")))
    if not epochs:
        hint = "avg" if prefer_avg else "non-avg"
        raise RuntimeError(f"No {hint} gen_ab checkpoints found in: {checkpoints_dir}")
    return max(epochs)


def _ckpt_path(checkpoints_dir: str, epoch: int, key: str) -> str:
    return os.path.join(checkpoints_dir, f"{epoch:04d}_net_{key}.pth")


def _style_state_paths(base_model_dir: str, epoch: int) -> Dict[str, str]:
    return {
        "epoch": os.path.join(base_model_dir, "checkpoints", f"{epoch:04d}_style_fusion_state.pth"),
        "final": os.path.join(base_model_dir, "style_fusion_state.pth"),
    }


def _load_state_dict_if_exists(module: torch.nn.Module, path: str, device: torch.device, strict: bool) -> bool:
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location=device)
    module.load_state_dict(state, strict=strict)
    return True


def _load_base_weights_into_model(
    model,
    base_checkpoints_dir: str,
    epoch: int,
    device: torch.device,
    *,
    strict: bool,
    init_from_avg: bool,
) -> None:
    """
    Load network weights from a base run into the current model.

    - Always tries to load discriminators and generators when present.
    - If init_from_avg is True and avg checkpoints exist, initializes gen_* from avg_gen_*.
    """
    base_checkpoints_dir = os.path.abspath(os.path.expanduser(base_checkpoints_dir))
    if not os.path.isdir(base_checkpoints_dir):
        raise FileNotFoundError(f"Base checkpoints dir not found: {base_checkpoints_dir}")

    # Generators.
    # If init_from_avg is requested, prefer avg weights as the initializer for gen_*.
    if init_from_avg:
        for gen_key in ("gen_ab", "gen_ba"):
            avg_key = f"avg_{gen_key}"
            avg_path = os.path.join(base_checkpoints_dir, f"{epoch:04d}_net_{avg_key}.pth")
            gen_path = os.path.join(base_checkpoints_dir, f"{epoch:04d}_net_{gen_key}.pth")
            loaded = False
            if hasattr(model.models, gen_key):
                if os.path.exists(avg_path):
                    loaded = _load_state_dict_if_exists(model.models[gen_key], avg_path, device, strict)
                if not loaded:
                    _load_state_dict_if_exists(model.models[gen_key], gen_path, device, strict)
            if hasattr(model.models, avg_key):
                _load_state_dict_if_exists(model.models[avg_key], avg_path, device, strict=False)
    else:
        for gen_key in ("gen_ab", "gen_ba", "avg_gen_ab", "avg_gen_ba"):
            if not hasattr(model.models, gen_key):
                continue
            path = _ckpt_path(base_checkpoints_dir, epoch, gen_key)
            _load_state_dict_if_exists(model.models[gen_key], path, device, strict=(strict and not gen_key.startswith("avg_")))

    # Discriminators (train-only).
    for disc_key in ("disc_a", "disc_b"):
        if not hasattr(model.models, disc_key):
            continue
        path = _ckpt_path(base_checkpoints_dir, epoch, disc_key)
        _load_state_dict_if_exists(model.models[disc_key], path, device, strict=strict)


def _maybe_load_style_fusion_state(model, base_model_dir: str, epoch: int) -> None:
    """
    Stylefusion model stores extra state (style_token_ba + count) outside net_*.pth.
    If the model supports it, load it from base epoch (fallback to final).
    """
    if not hasattr(model, "STYLE_FUSION_STATE_NAME"):
        return

    paths = _style_state_paths(base_model_dir, epoch)
    path = paths["epoch"] if os.path.exists(paths["epoch"]) else paths["final"]
    if not os.path.exists(path):
        print("[WARN] style_fusion_state not found in base run; style_token_ba will start empty.")
        return

    state = torch.load(path, map_location=model.device)
    token = state.get("style_token_ba", None)
    count = int(state.get("style_token_ba_count", 0))

    model.style_token_ba_count = count
    if token is None:
        model.style_token_ba = None
        return

    token = token.to(model.device)
    if token.ndim == 1:
        token = token.unsqueeze(0)
    model.style_token_ba = token
    print(f"[INFO] Loaded style_fusion_state from: {path}")


def _training_epoch(it_train, model, title: str, steps_per_epoch: Optional[int]) -> LossMetrics:
    model.train()
    steps = len(it_train)
    if steps_per_epoch is not None:
        steps = min(steps, int(steps_per_epoch))

    progbar = tqdm.tqdm(desc=title, total=steps, dynamic_ncols=True)
    metrics = LossMetrics()

    for batch in islice(it_train, steps):
        model.set_input(batch)
        model.optimization_step()
        metrics.update(model.get_current_losses())
        progbar.set_postfix(metrics.values, refresh=False)
        progbar.update()

    progbar.close()
    return metrics


def parse_cmdargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune from a checkpoint on a new dataset.")

    parser.add_argument("--base-checkpoints-dir", required=True, type=str, help="Path to base run checkpoints/ directory.")
    parser.add_argument(
        "--base-epoch",
        default="last",
        type=str,
        help="Base epoch to load (int) or 'last' (default).",
    )
    parser.add_argument(
        "--init-from-avg",
        action="store_true",
        default=False,
        help="Initialize gen_ab/gen_ba weights from avg_gen_* checkpoints when available.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=False,
        help="Allow partial state_dict loads (strict=False).",
    )

    # Dataset location options:
    #   - Either pass a single root directory (legacy behavior), which implies:
    #       root/BIT/trainA   (domain A adjacent-z pairs)
    #       root/FFPE_HE      (CycleGAN-style root containing trainB/testB for domain B)
    #   - OR pass explicit locations for the split directories you want to use:
    #       --trainA-path /abs/path/to/trainA
    #       --trainB-path /abs/path/to/trainB  (or the CycleGAN root containing trainB/)
    parser.add_argument(
        "--root-data-path",
        default=None,
        type=str,
        help="New dataset root (expects BIT/trainA and FFPE_HE under it).",
    )
    parser.add_argument(
        "--trainA-path",
        default=None,
        type=str,
        help="Exact path to domain A trainA directory (images for AdjacentZPairDataset).",
    )
    parser.add_argument(
        "--trainB-path",
        default=None,
        type=str,
        help=(
            "Exact path to domain B trainB directory OR the CycleGAN root containing trainB/. "
            "If you pass /.../trainB, the parent directory will be used as the CycleGAN root."
        ),
    )
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--checkpoint-every", default=10, type=int, help="Save checkpoints every N epochs.")
    parser.add_argument("--steps-per-epoch", default=2000, type=int)
    parser.add_argument("--num-workers", default=1, type=int)

    parser.add_argument("--z-spacing", default=2, type=int)
    parser.add_argument("--lambda-cycle", default=10.0, type=float)
    parser.add_argument("--lambda-gp", default=0.01, type=float)
    parser.add_argument("--lr-gen", default=5e-5, type=float)
    parser.add_argument("--lr-disc", default=1e-4, type=float)

    # Adjacent slice losses.
    parser.add_argument("--lambda-sub-loss", default=0.0, type=float)
    parser.add_argument("--lambda-embedding-loss", default=0.0, type=float)

    # Stylefusion extras.
    parser.add_argument("--lambda-style-fusion", default=0.0, type=float)
    parser.add_argument("--style-fusion-inject", choices=["add", "adain"], default="adain")
    parser.add_argument("--lambda-style-loss", default=1.0, type=float)

    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", *sorted(_KNOWN_MODELS)],
        help="Which model type to fine-tune. 'auto' uses the base run config.model.",
    )

    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        help=f"Output root (default: {ROOT_OUTDIR!r}).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        type=str,
        help="Optional run name appended under outdir (default: YYYYMMDD_finetune_<model>_from_epochXXXX).",
    )
    return parser.parse_args()


def main() -> None:
    cmd = parse_cmdargs()

    base_checkpoints_dir = os.path.abspath(os.path.expanduser(cmd.base_checkpoints_dir))
    base_model_dir = os.path.dirname(base_checkpoints_dir.rstrip(os.sep))

    base_args = Args.load(base_model_dir)
    base_model_name = str(base_args.config.model)

    if cmd.model == "auto":
        model_name = base_model_name
    else:
        model_name = str(cmd.model)

    if model_name not in _KNOWN_MODELS:
        raise RuntimeError(f"Unsupported model: {model_name}")

    if model_name != base_model_name:
        print(
            f"[WARN] Base config.model={base_model_name!r} but you requested model={model_name!r}. "
            "This will only work if the checkpoint weights are compatible."
        )

    if cmd.base_epoch == "last":
        epoch = _discover_last_epoch(base_checkpoints_dir, prefer_avg=bool(cmd.init_from_avg))
    else:
        epoch = int(cmd.base_epoch)

    # Load base config.json as a dict and apply overrides for the new dataset + training settings.
    base_config_dict: Dict[str, Any] = json.loads(base_args.config.to_json())

    def _normalize_cyclegan_root(path: str) -> str:
        path = os.path.abspath(os.path.expanduser(path))
        base = os.path.basename(path).lower()
        # If the user points directly to a split folder like ".../trainB",
        # use its parent as the CycleGAN root.
        if base in {"traina", "trainb", "testa", "testb", "vala", "valb"}:
            return os.path.dirname(path)
        return path

    # Resolve dataset locations.
    #
    # Domain A uses AdjacentZPairDataset and expects a directory of images (trainA).
    # Domain B uses the CycleGAN/ImageDomainFolder convention and expects a root
    # that contains trainB/ (and optionally testB/valB).
    if cmd.trainA_path or cmd.trainB_path:
        if not (cmd.trainA_path and cmd.trainB_path):
            raise SystemExit("If using explicit paths, both --trainA-path and --trainB-path are required.")
        data_path_a = os.path.abspath(os.path.expanduser(cmd.trainA_path))
        data_path_b = _normalize_cyclegan_root(cmd.trainB_path)
    else:
        if cmd.root_data_path is None:
            raise SystemExit("Either --root-data-path OR (--trainA-path and --trainB-path) must be provided.")
        root_data = os.path.abspath(os.path.expanduser(cmd.root_data_path))
        data_path_a = os.path.join(root_data, "BIT", "trainA")
        data_path_b = os.path.join(root_data, "FFPE_HE")

    # Preserve original dataset shapes/transforms when possible.
    ref_a = base_config_dict["data"]["datasets"][0]
    ref_b = base_config_dict["data"]["datasets"][1]

    dataset_config = [
        {
            "dataset": {
                "name": "adjacent-z-pairs",
                "domain": "A",
                "path": data_path_a,
                "z_spacing": int(cmd.z_spacing),
            },
            "shape": ref_a.get("shape", [3, 512, 512]),
            "transform_train": ref_a.get("transform_train", None),
            "transform_test": ref_a.get("transform_test", None),
        },
        {
            "dataset": {
                "name": "cyclegan",
                "domain": "B",
                "path": data_path_b,
            },
            "shape": ref_b.get("shape", [3, 512, 512]),
            "transform_train": ref_b.get("transform_train", None),
            "transform_test": ref_b.get("transform_test", None),
        },
    ]

    base_config_dict["batch_size"] = int(cmd.batch_size)
    base_config_dict["epochs"] = int(cmd.epochs)
    base_config_dict["steps_per_epoch"] = int(cmd.steps_per_epoch)
    base_config_dict["model"] = model_name
    base_config_dict["data"]["datasets"] = dataset_config
    base_config_dict["data"]["merge_type"] = "unpaired"
    base_config_dict["data"]["workers"] = int(cmd.num_workers)

    # Override optimizer LRs (keep betas etc from base config).
    if base_config_dict.get("generator", {}).get("optimizer"):
        base_config_dict["generator"]["optimizer"]["lr"] = float(cmd.lr_gen)
    if base_config_dict.get("discriminator", {}).get("optimizer"):
        base_config_dict["discriminator"]["optimizer"]["lr"] = float(cmd.lr_disc)

    # Override gradient penalty magnitude and cycle weight via model_args + gp config.
    if base_config_dict.get("gradient_penalty"):
        base_config_dict["gradient_penalty"]["lambda_gp"] = float(cmd.lambda_gp)

    model_args = dict(base_config_dict.get("model_args") or {})
    model_args["lambda_a"] = float(cmd.lambda_cycle)
    model_args["lambda_b"] = float(cmd.lambda_cycle)
    model_args["lambda_subtraction_loss"] = float(cmd.lambda_sub_loss)
    if model_name in {"uvcgan2_3D_embedding_loss", "uvcgan2_3D_stylefusion"}:
        model_args["lambda_embedding_loss"] = float(cmd.lambda_embedding_loss)
        model_args["z_spacing"] = float(cmd.z_spacing)
    if model_name == "uvcgan2_3D_subtraction_loss":
        model_args["z_spacing"] = float(cmd.z_spacing)
    if model_name == "uvcgan2_3D_stylefusion":
        model_args["lambda_style_fusion"] = float(cmd.lambda_style_fusion)
        model_args["style_fusion_inject"] = str(cmd.style_fusion_inject)
        model_args["lambda_style_loss"] = float(cmd.lambda_style_loss)

    base_config_dict["model_args"] = model_args

    # Output directory naming.
    today_str = date.today().strftime("%Y%m%d")
    if cmd.run_name is None:
        run_name = f"{today_str}_finetune_{model_name}_from_epoch{epoch:04d}"
    else:
        run_name = str(cmd.run_name)

    outdir = os.path.abspath(os.path.expanduser(cmd.outdir or ROOT_OUTDIR))
    outdir = os.path.join(outdir, run_name)

    args = Args.from_args_dict(
        outdir=outdir,
        label=None,
        log_level="DEBUG",
        checkpoint=int(cmd.checkpoint_every),
        **base_config_dict,
    )

    device = get_torch_device_smart()
    seed_everything(args.config.seed)

    # Build dataloader.
    it_train = construct_data_loaders(args.config.data, args.config.batch_size, split="train")

    # Construct and initialize model.
    model = construct_model(args.savedir, args.config, is_train=True, device=device)
    strict = not bool(cmd.allow_partial)

    _load_base_weights_into_model(
        model=model,
        base_checkpoints_dir=base_checkpoints_dir,
        epoch=epoch,
        device=device,
        strict=strict,
        init_from_avg=bool(cmd.init_from_avg),
    )
    _maybe_load_style_fusion_state(model, base_model_dir=base_model_dir, epoch=epoch)

    # Training history (CSV) will be written under args.savedir.
    history = TrainingHistory(args.savedir)

    for e in range(1, int(cmd.epochs) + 1):
        title = f"Fine-tune Epoch {e} / {cmd.epochs} (base epoch {epoch})"
        metrics = _training_epoch(it_train, model, title, steps_per_epoch=args.config.steps_per_epoch)

        history.end_epoch(e, metrics)
        model.end_epoch(e)

        if e % int(cmd.checkpoint_every) == 0:
            model.save(e)

    model.save(epoch=None)


if __name__ == "__main__":
    main()
