#!/usr/bin/env python

import argparse
import collections
import os

import tqdm
import numpy as np
from PIL import Image
import torch

from uvcgan2.consts import MERGE_NONE
from uvcgan2.eval.funcs import (
    load_eval_model_dset_from_cmdargs, tensor_to_image, slice_data_loader,
    get_eval_savedir, make_image_subdirs
)
from uvcgan2.utils.parsers import (
    add_standard_eval_parsers, add_plot_extension_parser
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Save model predictions as images'
    )
    add_standard_eval_parsers(parser)
    add_plot_extension_parser(parser)

    return parser.parse_args()

def save_images(model, savedir, filenames, ext):
    """Save model outputs using original filenames."""
    for (name, torch_image) in model.images.items():
        if torch_image is None:
            continue

        # model.images[name] is a batch of outputs: shape (N, C, H, W)
        for idx in range(torch_image.shape[0]):

            # ---- original filename corresponding to this output ----
            original_name = filenames[idx]

            # remove original extension, add new one later
            base = os.path.splitext(original_name)[0]

            # convert tensor → numpy uint8 image
            image = tensor_to_image(torch_image[idx])
            image = np.round(255 * image).astype(np.uint8)
            image = Image.fromarray(image)

            for e in ext:
                out_path = os.path.join(savedir, name, f"{base}.{e}")
                image.save(out_path)

def dump_single_domain_images(
    model, data_it, domain, n_eval, batch_size, savedir, sample_counter, ext
):
    # pylint: disable=too-many-arguments
    data_it, steps = slice_data_loader(data_it, batch_size, n_eval)
    desc = f'Translating domain {domain}'

    for batch in tqdm.tqdm(data_it, desc = desc, total = steps):
        #print(batch)
        # batch = [(tensor, filename), (tensor, filename), ...]
        #print(batch)
        images, names = batch

        # Convert list of tensors → batch tensor (N,C,H,W)
        images = torch.stack(images, dim=0)

        model.set_input(images, domain=domain)

        # and store filenames in model or return them later
        model.filenames = names

        torch.autograd.set_detect_anomaly(True)
        model.forward_nograd()

        save_images(model, savedir, names, ext)

def dump_images(model, data_list, n_eval, batch_size, savedir, ext):
    # pylint: disable=too-many-arguments
    make_image_subdirs(model, savedir)

    sample_counter = collections.defaultdict(int)
    if isinstance(ext, str):
        ext = [ ext, ]

    for domain, data_it in enumerate(data_list):
        dump_single_domain_images(
            model, data_it, domain, n_eval, batch_size, savedir,
            sample_counter, ext
        )

# Custom Collate Function
def inference_collate_fn(batch):
    # batch: List[(Tensor, str)] → ([Tensor, Tensor, ...], [str, str, ...])
    images = [item[0] for item in batch]
    names  = [item[1] for item in batch]
    return images, names

def main():
    cmdargs = parse_cmdargs()

    args, model, data_list, evaldir = load_eval_model_dset_from_cmdargs(
        cmdargs, merge_type = MERGE_NONE
    )

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
        cmdargs.ext
    )

if __name__ == '__main__':
    main() 

