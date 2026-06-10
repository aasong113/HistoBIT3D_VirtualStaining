import sys
import matplotlib.pyplot as plt
import numpy as np
import os

def set_two_domain_input(images, inputs, domain, device):
    if (domain is None) or (domain == 'both'):

        # Always define data_z from inputs[0]
        data_z = inputs[0]

        # This is for adjacent slice mode.
        if not isinstance(data_z, dict):
            images.real_a = data_z.to(device, non_blocking=True)

        else:
            z1_batch = data_z['z1']
            z2_batch = data_z['z2']

            images.real_a         = z1_batch.to(device, non_blocking=True)
            images.real_a_adj     = z2_batch.to(device, non_blocking=True)
            images.real_a_names   = data_z.get('z1_name', None)
            images.real_a_adj_names = data_z.get('z2_name', None)

        images.real_b = inputs[1].to(device, non_blocking=True)

    elif domain in ['a', 0]:
        images.real_a = inputs.to(device, non_blocking=True)

    elif domain in ['b', 1]:
        images.real_b = inputs.to(device, non_blocking=True)

    else:
        raise ValueError(
            f"Unknown domain: '{domain}'."
            " Supported domains: 'a' (alias 0), 'b' (alias 1), or 'both'"
        )
    
def save_image(tensor, filename):
    """
    Save a PyTorch tensor as an image. Supports both grayscale and RGB.

    Args:
        tensor (torch.Tensor): Inpt image tensor. Shape (H, W) for grayscale or (3, H, W) for RGB.
        filename (str): Output path to save the image.
    """
    tensor = tensor.detach().cpu()

    # Convert to NumPy
    if tensor.ndim == 2:
        # Grayscale image (H, W)
        image = tensor.numpy()
        plt.imsave(filename, image, cmap='gray')
    elif tensor.ndim == 3 and tensor.shape[0] == 3:
        # RGB image (3, H, W) → (H, W, 3)
        image = tensor.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)  # Optional: Clamp values for display
        plt.imsave(filename, image)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}. Expected (H, W) or (3, H, W).")


## handles regular image embedding and motion embedding visualization
def save_embedding_as_image(tensor, save_path, prefix="embed", max_channels=3):
    os.makedirs(save_path, exist_ok=True)
    tensor = tensor.detach().cpu()

    if tensor.ndim == 1:
        # This is already patch motion (no CLS)
        num_patches = tensor.shape[0]

        side = int(num_patches ** 0.5)
        assert side * side == num_patches, f"Cannot reshape {num_patches} tokens to square grid"

        image = tensor.reshape(side, side)
        save_image(image, os.path.join(save_path, f"{prefix}.png"))
        return

    elif tensor.ndim == 3:
        # Embedding: (N, B, C)
        N, B, C = tensor.shape
        assert B == 1
        patch_tokens = tensor[1:, 0, :]  # skip CLS → (1024, C)
        side = int((N - 1)**0.5)
        patch_grid = patch_tokens.T.reshape(C, side, side)  # (C, H, W)

        # Save average + first few channels
        save_image(patch_grid.mean(dim=0), os.path.join(save_path, f"{prefix}_mean.png"))
        for c in range(min(max_channels, C)):
            save_image(patch_grid[c], os.path.join(save_path, f"{prefix}_c{c}.png"))

    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")