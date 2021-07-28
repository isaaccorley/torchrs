import os
from glob import glob
from typing import Dict

import torch
import tifffile
import numpy as np
from PIL import Image

from torchrs.transforms import Compose, ToTensor


class S2Looking(torch.utils.data.Dataset):
    """ The Onera Satellite Change Detection (OSCD) dataset from 'Urban Change Detection for
    Multispectral Earth Observation Using Convolutional Neural Networks', Daudt at al. (2018)
    https://arxiv.org/abs/1703.00121

    'S2Looking is a building change detection dataset that contains large-scale side-looking
    satellite images captured at varying off-nadir angles. The S2Looking dataset consists of
    5,000 registered bitemporal image pairs (size of 1024*1024, 0.5 ~ 0.8 m/pixel) of rural
    areas throughout the world and more than 65,920 annotated change instances. We provide
    two label maps to separately indicate the newly built and demolished building regions
    for each sample in the dataset.'

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
        """ Returns a dict containing x, mask, region, dates
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
