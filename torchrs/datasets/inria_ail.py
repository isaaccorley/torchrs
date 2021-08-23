import os
import json
from glob import glob
from typing import List, Dict

import torch
import tifffile
import numpy as np

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

    def __init__(
        self,
        root: str = ".data/inria_ail",
        transform: Compose = Compose([ToTensor()]),
    ):
        self.root = root
        self.transform = transform
        self.images = self.load_images(self.image_root)

    @staticmethod
    def load_images(path: str) -> List[Dict]:
        pass

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        image_path, target_path = self.images[idx]["image"], self.images[idx]["target"]
        x, y = np.load(image_path), np.load(target_path)
        x, y = self.transform([x, y])
        return dict(x=x, mask=y)