#!/usr/bin/env python3
"""
Resume 3D UVCGAN training in-place from an existing model directory.

This script:
1) Loads config.json from an existing model directory.
2) Reuses the same outdir + label to target the exact same savedir.
3) Calls uvcgan2.train.train(), which auto-loads the last checkpoint epoch.
4) Uses a custom checkpoint cadence (default: every 5 epochs).
"""

import argparse
import json
import os
import re
import sys
from typing import Optional


def _find_latest_epoch(checkpoints_dir: str) -> int:
    if not os.path.isdir(checkpoints_dir):
        return 0

    pattern = re.compile(r"^(\d+)_net_gen_ab\.pth$")
    latest = 0

    for name in os.listdir(checkpoints_dir):
        match = pattern.match(name)
        if match is None:
            continue
        latest = max(latest, int(match.group(1)))

    return latest


def _read_label(model_dir: str) -> Optional[str]:
    label_path = os.path.join(model_dir, "label")
    if not os.path.isfile(label_path):
        return None

    with open(label_path, "r", encoding="utf-8") as f:
        label = f.read().strip()

    return label or None


def parse_args() -> argparse.Namespace:
    default_model_dir = (
        "/home/durrlab-asong/Anthony/3D_flow_consistent_UVCGANv2_vHE/outdir/"
        "20260210_Inverted_Combined_BIT2HE_duodenum_crypts_lieberkuhn_Style_injection/"
        "outdir/20260211_Inverted_Combined_BIT2HE_crypts_lieberkuhn_Style_injection_Train/"
        "20260211_duodenum_only_crypts_3DFlow_zspacing=2slices_lamsub=0p0_lamemb=0p0_lamSty=1p0/"
        "model_m(uvcgan2_3D_stylefusion)_d(basic)_g(vit-modnet)_"
        "uvcgan2-bn_(False:10.0:0.01:5e-05)"
    )

    parser = argparse.ArgumentParser(description="Resume 3D training from existing config/checkpoints.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=default_model_dir,
        help="Path to existing model directory containing config.json, label, checkpoints/.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional explicit config.json path (default: <model-dir>/config.json).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoints every N epochs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for total epochs (default: keep config value).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="Logging level passed into uvcgan2 train args.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings and exit without training.",
    )
    return parser.parse_args()


def main() -> None:
    cmd = parse_args()

    model_dir = os.path.abspath(os.path.expanduser(cmd.model_dir))
    config_path = cmd.config
    if config_path is None:
        config_path = os.path.join(model_dir, "config.json")
    config_path = os.path.abspath(os.path.expanduser(config_path))

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Ensure local repo import works even if launched from another directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if cmd.epochs is not None:
        config_dict["epochs"] = int(cmd.epochs)

    label = _read_label(model_dir)
    outdir = os.path.dirname(model_dir.rstrip(os.sep))
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    latest_epoch = _find_latest_epoch(checkpoints_dir)

    disc_model = (config_dict.get("discriminator") or {}).get("model")
    gen_model = (config_dict.get("generator") or {}).get("model")
    if label is not None:
        expected_name = (
            f"model_m({config_dict.get('model')})_d({disc_model})_g({gen_model})_{label}"
        ).replace("/", ":")
        expected_savedir = os.path.join(outdir, expected_name)
        if os.path.abspath(expected_savedir) != model_dir:
            raise RuntimeError(
                "Resolved savedir does not match requested model-dir.\n"
                f"  model-dir: {model_dir}\n"
                f"  resolved : {expected_savedir}\n"
                "Check label/config consistency before resuming."
            )

    args_dict = {
        **config_dict,
        "outdir": outdir,
        "label": label,
        "log_level": cmd.log_level,
        "checkpoint": int(cmd.checkpoint_every),
    }

    print("[INFO] Resume target")
    print(f"  model_dir         : {model_dir}")
    print(f"  config            : {config_path}")
    print(f"  label             : {label}")
    print(f"  outdir            : {outdir}")
    print(f"  last_checkpoint   : {latest_epoch}")
    print(f"  total_epochs      : {config_dict.get('epochs')}")
    print(f"  checkpoint_every  : {cmd.checkpoint_every}")

    if cmd.dry_run:
        print("[INFO] Dry run complete. No training started.")
        return

    from uvcgan2.train.train import train as run_train  # pylint: disable=import-error

    run_train(args_dict)


if __name__ == "__main__":
    main()
