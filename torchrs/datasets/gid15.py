import os
from glob import glob
from typing import List, Dict

import torch
import numpy as np
from PIL import Image

from torchrs.transforms import Compose, ToTensor


class GID15(torch.utils.data.Dataset):
    """ Gaofen Image Dataset (GID-15) from 'Land-Cover Classification with High-Resolution
    Remote Sensing Images Using Transferable Deep Models', Tong et al. (2018)
    https://arxiv.org/abs/1807.05713

    'We construct a new large-scale land-cover dataset with Gaofen-2 (GF-2) satellite
    images. This new dataset, which is named as Gaofen Image Dataset with 15 categories
    (GID-15), has superiorities over the existing land-cover dataset because of its
    large coverage, wide distribution, and high spatial resolution. The large-scale
    remote sensing semantic segmentation set contains 150 pixel-level annotated GF-2
    images, which is labeled in 15 categories.'
    """
    classes = [
        "background",
        "industrial_land",
        "urban_residential",
        "rural_residential",
        "traffic_land",
        "paddy_field",
        "irrigated_land",
        "dry_cropland",
        "garden_plot",
        "arbor_woodland",
        "shrub_land",
        "natural_grassland",
        "artificial_grassland",
        "river",
        "lake",
        "pond"
    ]
    splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = ".data/gid-15",
        split: str = "train",
        transform: Compose = Compose([ToTensor()]),
    ):
        self.split = split
        self.transform = transform
        self.images = self.load_images(os.path.join(root, "GID"), split)

    @staticmethod
    def load_images(path: str, split: str) -> List[Dict]:
        images = sorted(glob(os.path.join(path, "img_dir", split, "*.tif")))
        if split in ["train", "val"]:
            masks = [
                image.replace("img_dir", "ann_dir").replace(".tif", "_15label.png")
                for image in images
            ]
        else:
            masks = [None] * len(images)

        files = [dict(image=image, mask=mask) for image, mask in zip(images, masks)]
        return files

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        image_path, mask_path = self.images[idx]["image"], self.images[idx]["mask"]
        x = np.array(Image.open(image_path))

        if self.split in ["train", "val"]:
            y = np.array(Image.open(mask_path))
            x, y = self.transform([x, y])
            output = dict(x=x, mask=y)
        else:
            x = self.transform(x)
            output = dict(x=x)

        return output
