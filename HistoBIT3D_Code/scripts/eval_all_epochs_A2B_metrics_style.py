#!/usr/bin/env python3
"""
Evaluate a UVCGAN2(-3D) model across *all* checkpoint epochs (A->B inference + metrics),
with added support for the style-fusion model `uvcgan2_3D_emb_sub_stylefusion.py`
(registered as `uvcgan2_3D_stylefusion`) and its persisted style state file:

  - checkpoints/XXXX_style_fusion_state.pth
  - style_fusion_state.pth (final save, epoch=None)

Key behavior for style-fusion models:
  1) Loads the style token average automatically via model.load(epoch) (the model's
     _load_model_state reads the style_fusion_state file for that epoch).
  2) Optionally overrides which style_fusion_state is used via --style-fusion-state.
  3) Ensures style-injection hooks apply even when evaluating EMA/avg weights by:
       - copying avg_gen_{ab,ba} weights into gen_{ab,ba}
       - forcing avg_momentum=None so the forward uses gen_ab (where hooks live)

For non-style-fusion models, behavior matches eval_all_epochs_A2B_metrics.py.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
from typing import Dict, Optional, Sequence, Tuple

import torch

# Reuse all helper functions from the "base" eval script to keep behavior identical.
import eval_all_epochs_A2B_metrics as base  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run A->B inference for all epochs and compute PSNR/SSIM/LPIPS/FID/KID/IS "
            "(with optional style-fusion state loading for uvcgan2_3D_stylefusion)."
        )
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
        "--single-gpu",
        action="store_true",
        help="Disable DataParallel even if multiple GPUs are visible (use GPU 0 only).",
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
        "--style-fusion-inject",
        default=None,
        choices=["add", "adain"],
        help=(
            "Override the style injection method at inference time for uvcgan2_3D_stylefusion "
            "(default: use whatever the checkpoint/config specifies)."
        ),
        type=str,
    )
    parser.add_argument(
        "--lambda-style-fusion",
        default=None,
        help=(
            "Override lambda_style_fusion at inference time for uvcgan2_3D_stylefusion "
            "(default: use whatever the checkpoint/config specifies)."
        ),
        type=float,
    )
    parser.add_argument(
        "--style-fusion-state",
        default="auto",
        choices=["auto", "epoch", "final", "none"],
        help=(
            "Style-fusion state loading policy for uvcgan2_3D_stylefusion. "
            "'epoch' uses checkpoints/XXXX_style_fusion_state.pth (default behavior via model.load(epoch)); "
            "'final' forces style_fusion_state.pth (epoch=None) for every epoch; "
            "'auto' uses 'epoch' if available else falls back to 'final'; "
            "'none' disables style injection by clearing the loaded style token."
        ),
        type=str,
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


def _is_stylefusion_model(args) -> bool:
    # The style-fusion implementation lives in:
    #   uvcgan2/cgan/uvcgan2_3D_emb_sub_stylefusion.py
    # and is registered under this config.model name.
    return getattr(args.config, "model", None) == "uvcgan2_3D_stylefusion"


def _style_fusion_state_path(model_dir: str, epoch: Optional[int]) -> str:
    """
    Mirror uvcgan2.cgan.checkpoint.get_save_path(..., name="style_fusion_state", epoch=...).
    """
    if epoch is None:
        return os.path.join(model_dir, "style_fusion_state.pth")
    return os.path.join(model_dir, "checkpoints", f"{epoch:04d}_style_fusion_state.pth")


def _load_style_fusion_state_into_model(model, state_path: str) -> bool:
    """
    Manually load style_fusion_state into a model instance.
    This is only used for explicit overrides (--style-fusion-state=final/none/auto fallback).
    """
    if not os.path.exists(state_path):
        return False

    state = torch.load(state_path, map_location=model.device)
    token = state.get("style_token_ba", None)
    count = int(state.get("style_token_ba_count", 0))

    model.style_token_ba_count = count
    if token is None:
        model.style_token_ba = None
        return True

    token = token.to(model.device)
    if token.ndim == 1:
        token = token.unsqueeze(0)
    model.style_token_ba = token
    return True


def _apply_style_fusion_state_mode(
    model,
    model_dir: str,
    epoch: int,
    mode: str,
) -> Tuple[str, Optional[str]]:
    """
    Apply the requested style-fusion state policy and return (mode_used, path_used).
    """
    mode = str(mode)

    # Default behavior: model.load(epoch) already loaded the per-epoch state.
    if mode == "epoch":
        return "epoch", _style_fusion_state_path(model_dir, epoch)

    if mode == "none":
        model.style_token_ba = None
        return "none", None

    if mode == "final":
        path = _style_fusion_state_path(model_dir, None)
        ok = _load_style_fusion_state_into_model(model, path)
        return ("final", path if ok else None)

    # auto: keep epoch state if present; otherwise fall back to final.
    epoch_path = _style_fusion_state_path(model_dir, epoch)
    if os.path.exists(epoch_path):
        return "epoch", epoch_path

    final_path = _style_fusion_state_path(model_dir, None)
    ok = _load_style_fusion_state_into_model(model, final_path)
    return ("final", final_path if ok else None)


def _maybe_enable_avg_weights_for_stylefusion(model) -> None:
    """
    Style-fusion hooks are registered on gen_ab/gen_ba. If we want to evaluate EMA/avg
    weights, we copy avg_gen_* weights into gen_* and force avg_momentum=None.
    """
    if not (hasattr(model, "models") and hasattr(model.models, "avg_gen_ab")):
        return

    model.models.gen_ab.load_state_dict(model.models.avg_gen_ab.state_dict())
    model.models.gen_ba.load_state_dict(model.models.avg_gen_ba.state_dict())


def main() -> None:
    cmd = parse_args()

    if cmd.single_gpu:
        os.environ["UVCGAN2_SINGLE_GPU"] = "1"
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        print("[INFO] Single-GPU mode enabled: DataParallel disabled (using CUDA:0 when available).")

    checkpoints_dir = os.path.abspath(os.path.expanduser(cmd.checkpoints_dir))
    model_dir = os.path.dirname(checkpoints_dir.rstrip(os.sep))

    available_epochs = base._discover_epochs_from_checkpoints(checkpoints_dir, use_avg=cmd.use_avg)
    if not available_epochs:
        hint = "avg" if cmd.use_avg else "non-avg"
        extra = ""
        if not cmd.use_avg:
            extra = " (If you only have *_net_avg_gen_ab.pth files, re-run with --use-avg.)"
        raise RuntimeError(f"No {hint} epochs found in checkpoints dir: {checkpoints_dir}{extra}")

    epochs = base._parse_epochs_arg(cmd.epochs, available_epochs)
    if not epochs:
        raise RuntimeError("No epochs selected after applying --epochs filter.")

    output_dir = cmd.output_dir
    if output_dir is None:
        output_dir = os.path.join(model_dir, "eval_all_epochs_metrics")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    metrics_txt = os.path.join(output_dir, "metrics_by_epoch.txt")
    plot_path = os.path.join(output_dir, "metrics_by_epoch.png")
    samples_dir = os.path.join(output_dir, "samples_fake_b")
    os.makedirs(samples_dir, exist_ok=True)

    keep_best = not bool(cmd.no_keep_best)
    best_root = os.path.join(output_dir, "best_epochs")
    if keep_best:
        os.makedirs(best_root, exist_ok=True)
        print(f"[INFO] Keeping translated fake_b images for best epochs under: {best_root}")

    done_epochs = base._read_existing_epochs_from_txt(metrics_txt) if cmd.resume else set()
    if done_epochs:
        epochs = [e for e in epochs if e not in done_epochs]
        print(f"[INFO] Resume enabled: {len(done_epochs)} epochs already done; {len(epochs)} remaining.")
    if not epochs:
        print("[INFO] Nothing to do (all requested epochs already evaluated).")
        base._plot_metrics(base._load_metrics_table_txt(metrics_txt), plot_path)
        return

    # Validate paths.
    test_a_path = os.path.abspath(os.path.expanduser(cmd.test_a))
    real_b_dir = os.path.abspath(os.path.expanduser(cmd.real_b))
    if not os.path.exists(test_a_path):
        raise FileNotFoundError(f"--test-a not found: {test_a_path}")
    if not os.path.isdir(real_b_dir):
        raise FileNotFoundError(f"--real-b not found: {real_b_dir}")

    effective_split = cmd.split
    if cmd.dataset_name == "cyclegan":
        root, effective_split = base._resolve_cyclegan_root_and_split_for_test_a(test_a_path, cmd.split)
        if effective_split != cmd.split:
            print(
                f"[INFO] --test-a points to '{os.path.basename(test_a_path)}'; "
                f"overriding --split={cmd.split} -> {effective_split}"
            )
        test_a_images_dir = os.path.join(root, base._split_to_cyclegan_dirname(effective_split, "A"))
    else:
        test_a_images_dir = test_a_path

    test_a_files = base._list_image_files(test_a_images_dir, recursive=False)
    if not test_a_files:
        raise RuntimeError(f"No images found in inferred testA directory: {test_a_images_dir}")
    real_b_files = base._list_image_files(real_b_dir, recursive=False)
    if not real_b_files:
        raise RuntimeError(f"No images found in realB directory: {real_b_dir}")

    test_a_bases = [base._strip_known_image_suffixes(os.path.basename(p)) for p in test_a_files]
    sample_base = cmd.sample_basename or test_a_bases[0]
    if sample_base not in set(test_a_bases):
        raise RuntimeError(f"--sample-basename '{sample_base}' not found in testA filenames.")

    # Load model config once.
    args = base.Args.load(model_dir)
    device = base.get_torch_device_smart()
    base.seed_everything(args.config.seed)

    is_stylefusion = _is_stylefusion_model(args)

    model = base.construct_model(args.savedir, args.config, is_train=False, device=device)
    model.eval()

    # IMPORTANT: Style-fusion hooks are registered on gen_ab/gen_ba, not avg_gen_ab/avg_gen_ba.
    # So for style-fusion models we always force the forward to use gen_ab.
    if is_stylefusion and hasattr(model, "avg_momentum"):
        model.avg_momentum = None
    elif (not cmd.use_avg) and hasattr(model, "avg_momentum"):
        model.avg_momentum = None

    loader = base._configure_testA_only_dataloader(
        args=args,
        test_a_path=test_a_path,
        dataset_name=cmd.dataset_name,
        z_spacing=cmd.z_spacing,
        split=effective_split,
        batch_size=cmd.batch_size,
        num_workers=cmd.num_workers,
    )

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
        f.write(f"use_avg: {cmd.use_avg}\n")
        f.write(f"single_gpu: {cmd.single_gpu}\n")
        f.write(f"is_stylefusion_model: {is_stylefusion}\n")
        f.write(f"style_fusion_state_mode: {cmd.style_fusion_state}\n")
        f.write(f"style_fusion_inject_override: {cmd.style_fusion_inject}\n")
        f.write(f"lambda_style_fusion_override: {cmd.lambda_style_fusion}\n")

    print(f"[INFO] Writing metrics to: {metrics_txt}")
    print(f"[INFO] Saving sample fake_b images to: {samples_dir}")
    print(f"[INFO] Sample basename (constant across epochs): {sample_base}")

    best_epoch_for: Dict[str, Optional[int]] = {"fid": None, "kid": None, "is": None}
    best_value_for: Dict[str, Optional[float]] = {"fid": None, "kid": None, "is": None}
    best_metrics_by_epoch: Dict[int, set] = {}

    def _fmt_metric(x: float) -> str:
        return f"{float(x):.6f}" if base._is_finite_number(x) else "nan"

    for idx, epoch in enumerate(epochs, start=1):
        print(f"\n[INFO] Evaluating epoch {epoch} ({idx}/{len(epochs)})")

        model.load(epoch)
        model.eval()

        # Style-fusion model extras: ensure the style token average is loaded and,
        # if requested, override which style_fusion_state is used.
        if is_stylefusion:
            mode_used, path_used = _apply_style_fusion_state_mode(
                model=model,
                model_dir=model_dir,
                epoch=epoch,
                mode=cmd.style_fusion_state,
            )
            if path_used is not None:
                print(f"[INFO] style_fusion_state: {mode_used} ({path_used})")
            else:
                print(f"[WARN] style_fusion_state: {mode_used} (no file loaded)")

            # If the user asked for EMA inference, swap EMA weights into gen_* so
            # style-injection hooks still take effect.
            if cmd.use_avg:
                _maybe_enable_avg_weights_for_stylefusion(model)
            if hasattr(model, "avg_momentum"):
                model.avg_momentum = None

            # Optional inference-time overrides for style fusion behavior.
            # These are useful when you want to evaluate the same checkpoint under
            # different injection settings without re-training.
            if cmd.style_fusion_inject is not None:
                model.style_fusion_inject = str(cmd.style_fusion_inject)
            if cmd.lambda_style_fusion is not None:
                model.lambda_style_fusion_base = float(cmd.lambda_style_fusion)

        epoch_work_dir = os.path.join(output_dir, f"_tmp_epoch_{epoch:04d}")
        fake_b_dir = os.path.join(epoch_work_dir, "fake_b")
        os.makedirs(epoch_work_dir, exist_ok=True)

        base._translate_all_A_to_fakeB(
            model=model,
            loader=loader,
            out_fake_b_dir=fake_b_dir,
            n_eval=cmd.n_eval,
            ext="png",
        )

        sample_src = os.path.join(fake_b_dir, f"{sample_base}.png")
        if not os.path.exists(sample_src):
            raise RuntimeError(
                f"Sample output not found after inference: {sample_src}\n"
                "If your outputs are not PNG, change the script or set --sample-basename."
            )
        sample_dst = os.path.join(samples_dir, f"epoch_{epoch:04d}_{sample_base}.png")
        shutil.copyfile(sample_src, sample_dst)

        allow_unpaired = cmd.allow_unpaired or (not cmd.require_paired)

        num_pairs, psnr_val, ssim_val, lpips_val = base._compute_pairwise_metrics(
            fake_b_dir=fake_b_dir,
            real_b_dir=real_b_dir,
            allow_missing_metrics=cmd.allow_missing_metrics,
            resize_to_real=cmd.resize_to_real,
            allow_unpaired=allow_unpaired,
        )
        fid_val, kid_val, is_val = base._compute_distribution_metrics(
            fake_b_dir=fake_b_dir,
            real_b_dir=real_b_dir,
            allow_missing_metrics=cmd.allow_missing_metrics,
        )
        print(
            f"[METRICS][epoch {epoch:04d}] "
            f"FID={_fmt_metric(fid_val)}  "
            f"KID={_fmt_metric(kid_val)}  "
            f"IS={_fmt_metric(is_val)}"
        )

        row = base.EpochMetrics(
            epoch=epoch,
            num_pairs=num_pairs,
            psnr=psnr_val,
            ssim=ssim_val,
            lpips=lpips_val,
            fid=fid_val,
            kid=kid_val,
            inception_score=is_val,
        )

        base._append_metrics_row_txt(metrics_txt, row)
        base._plot_metrics(base._load_metrics_table_txt(metrics_txt), plot_path)

        # Keep fake_b images for best epochs (FID/KID/IS), matching the behavior
        # of the baseline eval script. This stores images under:
        #   <output_dir>/best_epochs/epoch_XXXX/fake_b/
        # and updates best_{fid,kid,is} symlinks/text pointers.
        if keep_best:
            improved = []
            for metric_key, value in (("fid", fid_val), ("kid", kid_val), ("is", is_val)):
                if not base._is_finite_number(value):
                    continue
                if base._metric_is_better(metric_key, float(value), best_value_for[metric_key]):
                    improved.append((metric_key, float(value)))

            if improved:
                epoch_kept_dir = os.path.join(best_root, f"epoch_{epoch:04d}")
                epoch_kept_fake_b = os.path.join(epoch_kept_dir, "fake_b")
                base._safe_move_dir(fake_b_dir, epoch_kept_fake_b)

                for metric_key, value in improved:
                    old_epoch = best_epoch_for[metric_key]
                    if old_epoch is not None:
                        old_set = best_metrics_by_epoch.get(old_epoch, set())
                        old_set.discard(metric_key)
                        best_metrics_by_epoch[old_epoch] = old_set
                        if not old_set:
                            base._safe_rmtree(os.path.join(best_root, f"epoch_{old_epoch:04d}"))

                    best_epoch_for[metric_key] = epoch
                    best_value_for[metric_key] = value
                    best_metrics_by_epoch.setdefault(epoch, set()).add(metric_key)
                    base._update_best_pointer(best_root, metric_key, epoch)

        # Cleanup: delete per-epoch scratch dir to save space.
        shutil.rmtree(epoch_work_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[INFO] Done.")
    print(f"[INFO] Metrics: {metrics_txt}")
    print(f"[INFO] Plot: {plot_path}")
    if keep_best:
        print(f"[INFO] Best epoch links: {best_root}")


if __name__ == "__main__":
    main()
