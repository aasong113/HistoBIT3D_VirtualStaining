# uvcgan2/data/adjacent_pair_dataset.py
import os
import glob
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
class AdjacentZPairDataset(Dataset):
    def __init__(self, root_dir, z_spacing=1, transform=None, **kwargs):
        self.root_dir = root_dir
        self.z_spacing = z_spacing
        self.transform = transform or transforms.ToTensor()
        print(f"[AdjacentZPairDataset] Scanning directory: {root_dir}")
        self.img_dict = self.build_img_dict()
        self.pairs = self.build_pairs()
        print(f"[AdjacentZPairDataset] Found {len(self.pairs)} valid Z-pairs")
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid adjacent Z pairs found in {root_dir}. "
                f"Check filename format and z_spacing={z_spacing}"
            )
    def build_img_dict(self):
        files = []
        for ext in ("*.tif", "*.tiff", "*.png"):
            files.extend(glob.glob(os.path.join(self.root_dir, ext)))
        print(f"[AdjacentZPairDataset] Found {len(files)} .tif/.tiff/.png files")
        pattern = re.compile(r"(.*)_img=(\d+)_P=(\d+)\.(tif|tiff|png)$", re.IGNORECASE)
        img_dict = defaultdict(list)
        for f in files:
            name = os.path.basename(f)
            match = pattern.search(name)  # IMPORTANT: search(), not match()
            if not match:
                print(f"[WARN] Filename did not match pattern: {name}")
                continue
            prefix = match.group(1)
            z = int(match.group(2))
            p = int(match.group(3))
            img_dict[(prefix, p)].append((z, f))
        # Sort each group by Z
        for key in img_dict:
            img_dict[key] = sorted(img_dict[key], key=lambda x: x[0])
        print(f"[AdjacentZPairDataset] Found {len(img_dict)} (prefix, P) groups")
        return img_dict
    def build_pairs(self):
        pairs = []
        for (prefix, p), z_files in self.img_dict.items():
            zs = [z for z, _ in z_files]
            if len(zs) <= self.z_spacing:
                continue
            for i in range(len(z_files) - self.z_spacing):
                z1, f1 = z_files[i]
                z2, f2 = z_files[i + self.z_spacing]
                pairs.append((f1, f2, z1, z2, prefix, p))
        return pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        f1, f2, z1, z2, prefix, p = self.pairs[idx]
        img1 = Image.open(f1).convert("RGB")
        img2 = Image.open(f2).convert("RGB")
        return {
            "z1": self.transform(img1),
            "z2": self.transform(img2),
            "z1_name": os.path.basename(f1),
            "z2_name": os.path.basename(f2),
            "meta": {
                "prefix": prefix,
                "patch_id": p,
                "z_t": z1,
                "z_t_plus": z2,
            },
        }
