import os
from glob import glob
from typing import List, Dict

import torch
import numpy as np
from PIL import Image

from torchrs.transforms import Compose, ToTensor


class ETCI2021(torch.utils.data.Dataset):
    """Sentinel-1 Synthetic Aperature Radar (SAR) segmentation dataset from the ETCI 2021 Competition on Flood Detection
    https://nasa-impact.github.io/etci2021/
    https://competitions.codalab.org/competitions/30440

    'The contest dataset is composed of 66,810 (33,405 x 2 VV & VH polarization) tiles of 256 x 256 pixels,
    distributed respectively across the training, validation and test sets as follows: 33,405, 10,400,
    and 12,348 tiles for each polarization. Each tile includes 3 RGB channels which have been converted
    by tiling 54 labeled GeoTiff files generated from Sentinel-1 C-band synthetic aperture radar (SAR)
    imagery data using Hybrid Pluggable Processing Pipeline hyp3.'

    Note that hyp3 preprocessing generates 3 band for each band so VV and VH are both of shape (256, 256, 3)
    """
    bands = ["VV", "VH"]
    splits = dict(train="train", val="test", test="test_internal")

    def __init__(
        self,
        root: str = ".data/etci2021",
        split: str = "train",
        transform: Compose = Compose([ToTensor()]),
    ):
        assert split in self.splits.keys()
        self.split = split
        self.transform = transform
        self.images = self.load_files(root, self.splits[split])

    @staticmethod
    def load_files(root: str, split: str) -> List[Dict]:
        images = []
        folders = sorted(glob(os.path.join(root, split, "*")))
        folders = [f + "/tiles" for f in folders]
        for folder in folders:
            vvs = glob(os.path.join(folder, "vv", "*.png"))
            vhs = glob(os.path.join(folder, "vh", "*.png"))
            water_masks = glob(os.path.join(folder, "water_body_label", "*.png"))

            if split == "test_internal":
                flood_masks = [None] * len(water_masks)
            else:
                flood_masks = glob(os.path.join(folder, "flood_label", "*.png"))

            for vv, vh, flood_mask, water_mask in zip(vvs, vhs, flood_masks, water_masks):
                images.append(dict(vv=vv, vh=vh, flood_mask=flood_mask, water_mask=water_mask))
        return images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """ Returns a dict containing vv, vh, flood mask, water mask
        vv: (3, h, w)
        vh: (3, h, w)
        flood mask: (1, h, w) flood mask
        water mask: (1, h, w) water mask
        """
        images = self.images[idx]
        vv = np.array(Image.open(images["vv"]), dtype="uint8")
        vh = np.array(Image.open(images["vh"]), dtype="uint8")
        water_mask = np.array(Image.open(images["water_mask"]).convert("L"), dtype="bool")

        if self.split == "test":
            vv, vh, water_mask = self.transform([vv, vh, water_mask])
            output = dict(vv=vv, vh=vh, water_mask=water_mask)
        else:
            flood_mask = np.array(Image.open(images["flood_mask"]).convert("L"), dtype="bool")
            vv, vh, flood_mask, water_mask = self.transform([vv, vh, flood_mask, water_mask])
            output = dict(vv=vv, vh=vh, flood_mask=flood_mask, water_mask=water_mask)

        return output
