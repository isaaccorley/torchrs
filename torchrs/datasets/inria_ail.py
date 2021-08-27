import os
import re
from glob import glob
from typing import List, Dict

import torch
import numpy as np
from PIL import Image

from torchrs.transforms import Compose, ToTensor


class InriaAIL(torch.utils.data.Dataset):
    """ Inria Aerial Image Labeling dataset from 'Can semantic labeling methods
    generalize to any city? the inria aerial image labeling benchmark', Maggiori et al. (2017)
    https://ieeexplore.ieee.org/document/8127684

    'The training set contains 180 color image tiles of size 5000x5000, covering a surface of 1500mx1500m
    each (at a 30 cm resolution). There are 36 tiles for each of the following regions (Austin, Chicago, Kitsap County, Western Tyrol, Vienna)
    The format is GeoTIFF. Files are named by a prefix associated to the region (e.g., austin- or vienna-)
    followed by the tile number (1-36). The reference data is in a different folder and the file names
    correspond exactly to those of the color images. In the case of the reference data, the tiles are
    single-channel images with values 255 for the building class and 0 for the not building class.'
    """
    splits = ["train", "test"]

    def __init__(
        self,
        root: str = ".data/AerialImageDataset",
        split: str = "train",
        transform: Compose = Compose([ToTensor()]),
    ):
        self.split = split
        self.transform = transform
        self.images = self.load_images(root, split)
        self.regions = sorted(list(set(image["region"] for image in self.images)))

    @staticmethod
    def load_images(path: str, split: str) -> List[Dict]:
        images = sorted(glob(os.path.join(path, split, "images", "*.tif")))
        pattern = re.compile("[a-zA-Z]+")
        regions = [re.findall(pattern, os.path.basename(image))[0] for image in images]

        if split == "train":
            targets = sorted(glob(os.path.join(path, split, "gt", "*.tif")))
        else:
            targets = [None] * len(images)

        files = [
            dict(image=image, target=target, region=region)
            for image, target, region in zip(images, targets, regions)
        ]
        return files

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        image_path, target_path = self.images[idx]["image"], self.images[idx]["target"]
        x = np.array(Image.open(image_path))

        if self.split == "train":
            y = np.array(Image.open(target_path))
            y = np.clip(y, a_min=0, a_max=1)
            x, y = self.transform([x, y])
            output = dict(x=x, mask=y, region=self.images[idx]["region"])
        else:
            x = self.transform(x)
            output = dict(x=x)

        return output
