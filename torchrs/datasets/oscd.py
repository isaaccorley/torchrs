import os
from glob import glob
from typing import Dict

import torch
import tifffile
import numpy as np
from PIL import Image

from torchrs.transforms import Compose, ToTensor


def sort(x):
    band = os.path.splitext(x.split(os.sep)[-1])[0]
    if band == "B8A":
        band = "B08A"
    return band


class OSCD(torch.utils.data.Dataset):
    """ The Onera Satellite Change Detection (OSCD) dataset from 'Urban Change Detection for
    Multispectral Earth Observation Using Convolutional Neural Networks', Daudt at al. (2018)
    https://arxiv.org/abs/1703.00121

    'The Onera Satellite Change Detection dataset addresses the issue of detecting changes between
    satellite images from different dates. It comprises 24 pairs of multispectral images taken
    from the Sentinel-2 satellites between 2015 and 2018. Locations are picked all over the world,
    in Brazil, USA, Europe, Middle-East and Asia. For each location, registered pairs of 13-band
    multispectral satellite images obtained by the Sentinel-2 satellites are provided. Images vary
    in spatial resolution between 10m, 20m and 60m. Pixel-level change ground truth is provided for
    all 14 training and 10 test image pairs. The annotated changes focus on urban changes, such as
    new buildings or new roads. These data can be used for training and setting parameters of change
    detection algorithms.
    """
    def __init__(
        self,
        root: str = ".data/oscd",
        split: str = "train",
        transform: Compose = Compose([ToTensor(permute_dims=False)]),
    ):
        assert split in ["train", "test"]
        self.root = root
        self.transform = transform
        self.regions = self.load_files(root, split)

    @staticmethod
    def load_files(root: str, split: str):
        regions = []
        labels_root = os.path.join(root, f"{split}_labels")
        images_root = os.path.join(root, "images")
        folders = glob(os.path.join(labels_root, "*/"))
        for folder in folders:
            region = folder.split(os.sep)[-2]
            mask = os.path.join(labels_root, region, "cm", "cm.png")
            images1 = glob(os.path.join(images_root, region, "imgs_1_rect", "*.tif"))
            images2 = glob(os.path.join(images_root, region, "imgs_2_rect", "*.tif"))
            images1 = sorted(images1, key=sort)
            images2 = sorted(images2, key=sort)
            with open(os.path.join(images_root, region, "dates.txt")) as f:
                dates = tuple([line.split()[-1] for line in f.read().strip().splitlines()])

            regions.append(dict(region=region, images1=images1, images2=images2, mask=mask, dates=dates))

        return regions

    def __len__(self) -> int:
        return len(self.regions)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, mask
        x: (2, 13, h, w)
        mask: (1, h, w)
        """
        region = self.regions[idx]
        mask = np.array(Image.open(region["mask"]))
        mask[mask == 255] = 1
        image1 = np.stack([tifffile.imread(path) for path in region["images1"]], axis=0)
        image2 = np.stack([tifffile.imread(path) for path in region["images2"]], axis=0)
        image1, image2, mask = self.transform([image1, image2, mask])
        x = torch.stack([image1, image2], dim=0)
        return dict(x=x, mask=mask)
