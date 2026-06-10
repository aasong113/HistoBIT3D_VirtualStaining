#!/usr/bin/env python

import argparse
import collections
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
    start_model_eval, tensor_to_image, slice_data_loader, get_eval_savedir,
    make_image_subdirs
)
from uvcgan2.data import construct_data_loaders
from uvcgan2.config.data_config import DatasetConfig
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

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Save model predictions as images'
    )
    add_standard_eval_parsers(parser, default_epoch = 10)
    add_plot_extension_parser(parser)

    parser.add_argument(
        '--test_data_path',
        '--data-path',
        '--data_path',
        default = None,
        dest    = 'test_data_path',
        help    = (
            'Path to test images for A->B translation only. Can be either the '
            'CycleGAN root (containing testA/) or the testA/ directory itself.'
        ),
        type    = str,
    )

    parser.add_argument(
        '--data-name',
        '--dataset-name',
        default = 'cyclegan',
        dest    = 'dataset_name',
        choices = [ 'cyclegan', 'adjacent-z-pairs' ],
        help    = 'dataset type to use for inference',
        type    = str,
    )

    parser.add_argument(
        '--z-spacing',
        default = 1,
        dest    = 'z_spacing',
        help    = 'z spacing (only for adjacent-z-pairs)',
        type    = int,
    )

    parser.add_argument(
        '--debug-save-input',
        action  = 'store_true',
        default = False,
        dest    = 'debug_save_input',
        help    = 'save a few raw input images before translation',
    )
    parser.add_argument(
        '--debug-max-batches',
        default = 2,
        dest    = 'debug_max_batches',
        help    = 'max batches to debug-save/print',
        type    = int,
    )
    parser.add_argument(
        '--debug-autoscale',
        action  = 'store_true',
        default = False,
        dest    = 'debug_autoscale',
        help    = 'also save autoscaled input images (min-max per-image)',
    )
    parser.add_argument(
        '--debug-print-stats',
        action  = 'store_true',
        default = False,
        dest    = 'debug_print_stats',
        help    = 'print basic tensor stats for inputs',
    )
    parser.add_argument(
        '--debug-print-output-stats',
        action  = 'store_true',
        default = False,
        dest    = 'debug_print_output_stats',
        help    = 'print basic tensor stats for outputs after translation',
    )

    return parser.parse_args()

def _infer_cyclegan_root(path):
    path = os.path.abspath(os.path.expanduser(path))
    base = os.path.basename(path).lower()
    if base in {'testa', 'traina', 'vala', 'testb', 'trainb', 'valb'}:
        return os.path.dirname(path)
    return path

def _configure_a_to_b_test_dataloader(
    args, test_data_path, dataset_name, z_spacing, split
):
    dataset_name = str(dataset_name)
    test_data_path = os.path.abspath(os.path.expanduser(test_data_path))

    if dataset_name == 'cyclegan':
        path = _infer_cyclegan_root(test_data_path)
    elif dataset_name == 'adjacent-z-pairs':
        path = test_data_path
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    ref = args.config.data.datasets[0]
    shape = tuple(ref.shape) if isinstance(ref.shape, (list, tuple)) else ref.shape

    dataset_args = {
        'name'   : dataset_name,
        'domain' : 'A',
        'path'   : path,
    }
    if dataset_name == 'adjacent-z-pairs':
        dataset_args['z_spacing'] = int(z_spacing)

    args.config.data.datasets = [
        DatasetConfig(
            dataset = dataset_args,
            shape = shape,
            transform_train = None,
            transform_test  = ref.transform_test,
        )
    ]
    args.config.data.merge_type = MERGE_NONE

def _tensor_stats(tensor):
    tensor = tensor.detach()
    return {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype).replace('torch.', ''),
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std(unbiased=False).item()),
    }

def _tensor_to_pil_image(tensor, autoscale=False):
    image = tensor_to_image(tensor)
    if autoscale:
        vmin = float(np.min(image))
        vmax = float(np.max(image))
        denom = max(vmax - vmin, 1e-8)
        image = (image - vmin) / denom
    else:
        image = np.clip(image, 0.0, 1.0)

    image = np.round(255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return Image.fromarray(image)

def _iter_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
        return

    if isinstance(obj, (list, tuple)):
        for x in obj:
            if isinstance(x, torch.Tensor):
                yield x

def save_images(model, savedir, filenames, ext):
    """Save model outputs using original filenames."""
    for (name, torch_image) in model.images.items():
        if torch_image is None:
            continue

        # model.images[name] is a batch of outputs: shape (N, C, H, W)
        for idx in range(torch_image.shape[0]):

            # ---- original filename corresponding to this output ----
            original_name = filenames[idx]

            # strip any known image suffixes (handles e.g. ".tif.tif")
            base = _strip_known_image_suffixes(original_name)

            # convert tensor → numpy uint8 image
            image = tensor_to_image(torch_image[idx])
            image = np.round(255 * image).astype(np.uint8)
            image = Image.fromarray(image)

            for e in ext:
                out_path = os.path.join(savedir, name, f"{base}.{e}")
                image.save(out_path)

def dump_single_domain_images(
    model,
    data_it,
    domain,
    n_eval,
    batch_size,
    savedir,
    sample_counter,
    ext,
    debug_dir = None,
    debug_max_batches = 0,
    debug_autoscale = False,
    debug_print_stats = False,
    debug_print_output_stats = False,
):
    # pylint: disable=too-many-arguments
    data_it, steps = slice_data_loader(data_it, batch_size, n_eval)
    desc = f'Translating domain {domain}'

    for batch_idx, batch in enumerate(tqdm.tqdm(data_it, desc = desc, total = steps)):
        #print(batch)
        # batch = [(tensor, filename), (tensor, filename), ...]
        #print(batch)
        images, names = batch

        if debug_print_stats and batch_idx < debug_max_batches:
            try:
                stats = _tensor_stats(images[0])
                print(f"[DEBUG] input[{domain}] {names[0]}: {stats}")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[DEBUG] failed to compute input stats: {exc}")

        if debug_dir is not None and batch_idx < debug_max_batches:
            os.makedirs(debug_dir, exist_ok=True)
            try:
                base0 = os.path.splitext(str(names[0]))[0]
                pil = _tensor_to_pil_image(images[0], autoscale=False)
                out_path = os.path.join(
                    debug_dir,
                    f"domain{domain}_batch{batch_idx:04d}_input_raw_{base0}.png",
                )
                pil.save(out_path)
                if debug_autoscale:
                    pil2 = _tensor_to_pil_image(images[0], autoscale=True)
                    out_path2 = os.path.join(
                        debug_dir,
                        f"domain{domain}_batch{batch_idx:04d}_input_autoscale_{base0}.png",
                    )
                    pil2.save(out_path2)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[DEBUG] failed to save input debug image: {exc}")

        # Convert list of tensors → batch tensor (N,C,H,W)
        images = torch.stack(images, dim=0)

        model.set_input(images, domain=domain)

        # and store filenames in model or return them later
        model.filenames = names

        torch.autograd.set_detect_anomaly(True)
        model.forward_nograd()

        if debug_print_output_stats and batch_idx < debug_max_batches:
            try:
                for out_name, out_val in model.images.items():
                    for out_tensor in _iter_tensors(out_val):
                        stats = _tensor_stats(out_tensor[0])
                        print(f"[DEBUG] output[{domain}] {out_name}: {stats}")
                        break
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[DEBUG] failed to compute output stats: {exc}")

        save_images(model, savedir, names, ext)

def dump_images(
    model,
    data_list,
    n_eval,
    batch_size,
    savedir,
    ext,
    debug_dir = None,
    debug_max_batches = 0,
    debug_autoscale = False,
    debug_print_stats = False,
    debug_print_output_stats = False,
):
    # pylint: disable=too-many-arguments
    make_image_subdirs(model, savedir)

    sample_counter = collections.defaultdict(int)
    if isinstance(ext, str):
        ext = [ ext, ]

    for domain, data_it in enumerate(data_list):
        dump_single_domain_images(
            model, data_it, domain, n_eval, batch_size, savedir,
            sample_counter, ext,
            debug_dir = debug_dir,
            debug_max_batches = debug_max_batches,
            debug_autoscale = debug_autoscale,
            debug_print_stats = debug_print_stats,
            debug_print_output_stats = debug_print_output_stats,
        )

# Custom Collate Function
def inference_collate_fn(batch):
    first = batch[0]

    # CycleGAN/image folder datasets in inference mode:
    #   (tensor, filename)
    if isinstance(first, (list, tuple)) and len(first) == 2:
        images = [item[0] for item in batch]
        names  = [item[1] for item in batch]
        return images, names

    # AdjacentZPairDataset returns dict:
    #   { "z1": tensor, "z2": tensor, "z1_name": str, "z2_name": str, ... }
    if isinstance(first, dict):
        images = []
        names = []
        for item in batch:
            if 'z1' in item:
                images.append(item['z1'])
                names.append(item.get('z1_name', ''))
            if 'z2' in item:
                images.append(item['z2'])
                names.append(item.get('z2_name', ''))
        return images, names

    raise TypeError(f"Unsupported batch item type: {type(first)}")

def main():
    cmdargs = parse_cmdargs()

    args, model, evaldir = start_model_eval(
        cmdargs.model,
        cmdargs.epoch,
        cmdargs.model_state,
        merge_type = MERGE_NONE,
        batch_size = cmdargs.batch_size,
    )

    if cmdargs.test_data_path:
        _configure_a_to_b_test_dataloader(
            args,
            cmdargs.test_data_path,
            cmdargs.dataset_name,
            cmdargs.z_spacing,
            cmdargs.split,
        )

    data_list = construct_data_loaders(
        args.config.data, args.config.batch_size, split = cmdargs.split
    )
    if isinstance(data_list, (list, tuple)) and len(data_list) > 1:
        data_list = [ data_list[0] ]

    # Set inference mode + patch DataLoader(s) with custom collate_fn
    if isinstance(data_list, (list, tuple)):
        new_list = []
        for dl in data_list:
            ds = dl.dataset
            if hasattr(ds, 'set_inference'):
                ds.set_inference(True)

            new_dl = torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=dl.num_workers if hasattr(dl, "num_workers") else 0,
                collate_fn=inference_collate_fn
            )
            new_list.append(new_dl)
        data_list = new_list
    else:
        ds = data_list.dataset
        if hasattr(ds, 'set_inference'):
            ds.set_inference(True)

        data_list = [
            torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=data_list.num_workers if hasattr(data_list, "num_workers") else 0,
                collate_fn=inference_collate_fn
            )
        ]

    savedir = get_eval_savedir(
        evaldir, 'images', cmdargs.model_state, cmdargs.split
    )

    dump_images(
        model, data_list, cmdargs.n_eval, args.batch_size, savedir,
        cmdargs.ext,
        debug_dir = None if not cmdargs.debug_save_input else os.path.join(savedir, '_debug_inputs'),
        debug_max_batches = cmdargs.debug_max_batches,
        debug_autoscale = cmdargs.debug_autoscale,
        debug_print_stats = cmdargs.debug_print_stats,
        debug_print_output_stats = cmdargs.debug_print_output_stats,
    )

if __name__ == '__main__':
    main() 
