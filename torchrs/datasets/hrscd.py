import os
from glob import glob
from typing import List, Dict

import torch
import tifffile

from torchrs.transforms import Compose, ToTensor


class HRSCD(torch.utils.data.Dataset):
    """ The High Resolution Semantic Change Detection (HRSCD) dataset from 'Multitask Learning
    for Large-scale Semantic Change Detection', Daudt at al. (2018)
    https://arxiv.org/abs/1810.08452

    'This dataset contains 291 coregistered image pairs of RGB aerial images from IGS's
    BD ORTHO database. Pixel-level change and land cover annotations are provided, generated
    by rasterizing Urban Atlas 2006, Urban Atlas 2012, and Urban Atlas Change 2006-2012 maps.'
    """
    classes = [
        "No Information",
        "Artificial surfaces",
        "Agricultural areas",
        "Forests",
        "Wetlands",
        "Water"
    ]

    def __init__(
        self,
        root: str = ".data/HRSCD",
        transform: Compose = Compose([ToTensor()])
    ):
        self.root = root
        self.transform = transform
        self.files = self.load_files(root)

    @staticmethod
    def load_files(root: str) -> List[Dict]:
        images1 = sorted(glob(os.path.join(root, "images", "2006", "**", "*.tif"), recursive=True))
        images2 = sorted(glob(os.path.join(root, "images", "2012", "**", "*.tif"), recursive=True))
        lcs1 = sorted(glob(os.path.join(root, "labels", "2006", "**", "*.tif"), recursive=True))
        lcs2 = sorted(glob(os.path.join(root, "labels", "2012", "**", "*.tif"), recursive=True))
        changes = sorted(glob(os.path.join(root, "labels", "change", "**", "*.tif"), recursive=True))
        files = []
        for image1, image2, lc1, lc2, change in zip(images1, images2, lcs1, lcs2, changes):
            region = image1.split(os.sep)[-2]
            files.append(dict(image1=image1, image2=image2, lc1=lc1, lc2=lc2, mask=change, region=region))
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, land cover mask, change mask
        x: (2, 3, 1000, 1000)
        lc: (2, 1000, 1000)
        mask: (1, 1000, 1000)
        """
        files = self.files[idx]
        image1 = tifffile.imread(files["image1"])
        image2 = tifffile.imread(files["image2"])
        lc1 = tifffile.imread(files["lc1"])
        lc2 = tifffile.imread(files["lc2"])
        mask = tifffile.imread(files["mask"])
        image1, image2, lc1, lc2, mask = self.transform([image1, image2, lc1, lc2, mask])
        x = torch.stack([image1, image2], dim=0)
        lc = torch.cat([lc1, lc2], dim=0)
        return dict(x=x, lc=lc, mask=mask)
