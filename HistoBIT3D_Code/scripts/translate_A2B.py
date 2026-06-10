#!/usr/bin/env python

import argparse
import os
import sys

import tqdm
import numpy as np
from PIL import Image
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from uvcgan2.consts import MERGE_NONE
from uvcgan2.eval.funcs import (
    start_model_eval, tensor_to_image, slice_data_loader, get_eval_savedir
)
from uvcgan2.config.data_config import DatasetConfig
from uvcgan2.data import construct_data_loaders
from uvcgan2.utils.parsers import (
    add_standard_eval_parsers, add_plot_extension_parser
)

_KNOWN_IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}


def _strip_known_image_suffixes(name):
    base = os.path.basename(str(name))
    while True:
        root, ext = os.path.splitext(base)
        if ext.lower() in _KNOWN_IMAGE_EXTS and root:
            base = root
            continue
        return base


def _infer_cyclegan_root(path):
    path = os.path.abspath(os.path.expanduser(path))
    base = os.path.basename(path).lower()
    if base in {'testa', 'traina', 'vala', 'testb', 'trainb', 'valb'}:
        return os.path.dirname(path)
    return path


def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description='A->B translation using training-time dataloader params'
    )
    add_standard_eval_parsers(parser, default_epoch=200)
    add_plot_extension_parser(parser)

    parser.add_argument(
        '--data-root',
        '--data_root',
        default='/home/durrlab-asong/Anthony/subset_training_data_crypts/BIT/testA',
        dest='data_root',
        help='Path to images to translate (root containing testA/ or the testA/ directory itself).',
        type=str,
    )

    parser.add_argument(
        '--n',
        default=None,
        dest='n_eval',
        help='Number of images to translate (default: all).',
        type=int,
    )

    parser.add_argument(
        '--use-avg',
        action='store_true',
        default=False,
        dest='use_avg',
        help='Use avg_gen_ab if present (EMA), matching eval-time behavior.',
    )

    return parser.parse_args()


def _configure_a_only_dataset(args, data_root):
    data_root = os.path.abspath(os.path.expanduser(data_root))
    cyclegan_root = _infer_cyclegan_root(data_root)

    # Preserve training-time shape + transforms from the original domain A dataset.
    ref = args.config.data.datasets[0]
    shape = tuple(ref.shape) if isinstance(ref.shape, (list, tuple)) else ref.shape

    args.config.data.datasets = [
        DatasetConfig(
            dataset={
                'name': 'cyclegan',
                'domain': 'A',
                'path': cyclegan_root,
            },
            shape=shape,
            transform_train=None,
            transform_test=ref.transform_test,
        )
    ]
    args.config.data.merge_type = MERGE_NONE


def _save_tensor_image(tensor, out_path):
    image = tensor_to_image(tensor)
    image = np.clip(image, 0.0, 1.0)
    image = np.round(255 * image).astype(np.uint8)
    Image.fromarray(image).save(out_path)


def main():
    cmdargs = parse_cmdargs()

    args, model, evaldir = start_model_eval(
        cmdargs.model,
        cmdargs.epoch,
        cmdargs.model_state,
        merge_type=MERGE_NONE,
        batch_size=cmdargs.batch_size,
    )

    _configure_a_only_dataset(args, cmdargs.data_root)

    loader = construct_data_loaders(
        args.config.data, args.config.batch_size, split=cmdargs.split
    )

    # Ensure filename-returning behavior for CycleGAN-style datasets.
    if hasattr(loader.dataset, 'set_inference'):
        loader.dataset.set_inference(True)

    savedir = get_eval_savedir(
        evaldir, 'temp_ab', cmdargs.model_state, cmdargs.split, mkdir=True
    )
    real_dir = os.path.join(savedir, 'real_a')
    fake_dir = os.path.join(savedir, 'fake_b')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    print(f"Saving A->B translation outputs to: {savedir}")
    print(f"  inputs:  {real_dir}")
    print(f"  outputs: {fake_dir}")

    gen_ab = None
    if hasattr(model, 'models'):
        if cmdargs.use_avg and hasattr(model.models, 'avg_gen_ab'):
            gen_ab = model.models.avg_gen_ab
        elif hasattr(model.models, 'gen_ab'):
            gen_ab = model.models.gen_ab

    if gen_ab is None:
        raise RuntimeError("Could not find generator 'gen_ab' on the loaded model.")

    data_it, steps = slice_data_loader(loader, cmdargs.batch_size, cmdargs.n_eval)
    desc = 'Translating A->B'

    for images, names in tqdm.tqdm(data_it, desc=desc, total=steps):
        # Default collate returns (tensor_batch, [str, ...]) here.
        if isinstance(names, str):
            names = [names]

        images = images.to(model.device, non_blocking=True)
        with torch.no_grad():
            fake_b = gen_ab(images)

        for i, name in enumerate(names):
            base = _strip_known_image_suffixes(name)
            for e in (cmdargs.ext or ['png']):
                _save_tensor_image(images[i].detach().cpu(), os.path.join(real_dir, f'{base}.{e}'))
                _save_tensor_image(fake_b[i].detach().cpu(), os.path.join(fake_dir, f'{base}.{e}'))


if __name__ == '__main__':
    main()
