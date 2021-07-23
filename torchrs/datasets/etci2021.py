import os
from glob import glob
from typing import List, Tuple, Dict

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

from torchrs.transforms import ToTensor


class PROBAV(torch.utils.data.Dataset):
    """Sentinel-1 Synthetic Aperature Radar (SAR) segmentation dataset from the ETCI 2021 Competition on Flood Detection
    https://nasa-impact.github.io/etci2021/
    https://competitions.codalab.org/competitions/30440

    'The contest dataset is composed of 66,810 (33,405 x 2 VV & VH polarization) tiles of 256 x 256 pixels,
    distributed respectively across the training, validation and test sets as follows: 33,405, 10,400,
    and 12,348 tiles for each polarization. Each tile includes 3 RGB channels which have been converted
    by tiling 54 labeled GeoTiff files generated from Sentinel-1 C-band synthetic aperture radar (SAR)
    imagery data using Hybrid Pluggable Processing Pipeline hyp3.'
    """
    bands = ["VV", "VH"]
    splits = dict(train="train", val="testing", test="testing_internal")

    def __init__(
        self,
        root: str =".data/etci2021",
        split: str = "train",
        transform: T.Compose = T.Compose([ToTensor()]),
    ):
        assert split in self.splits.keys()
        self.transform = transform
        self.images = self.load_files(root, split)

    @staticmethod
    def load_files(root: str, split: str) -> List[Dict]:
        imgsets = []
        folders = sorted(glob(os.path.join(root, split, band, "imgset*")))
        for folder in folders:
            lr = sorted(glob(os.path.join(folder, "LR*.png")))
            qm = sorted(glob(os.path.join(folder, "QM*.png")))
            sm = glob(os.path.join(folder, "SM.png"))[0]
            hr = glob(os.path.join(folder, "HR.png"))[0]
            imgsets.append(dict(lr=lr, qm=qm, hr=hr, sm=sm))
        return imgsets

    def len(self) -> int:
        return len(self.imgsets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns a dict containing lrs, qms, hr, sm
        lrs: (t, h, w) low resolution images
        qms: (t, h, w) low resolution image quality masks
        hr: (h, w) high resolution image
        sm: (h, w) high resolution image status mask

        Note: 
        lr/qm original size is (128, 128),
        hr/sm original size is (384, 384) (scale factor = 3)
        t is the number of lr images for an image set (min = 9)
        """
        imgset = self.imgsets[idx]

        # Load
        lrs = [np.array(Image.open(lr), dtype="int32") for lr in imgset["lr"]]
        qms = [np.array(Image.open(qm), dtype="bool") for qm in imgset["qm"]]
        hr = np.array(Image.open(imgset["hr"]), dtype="int32")
        sm = np.array(Image.open(imgset["sm"]), dtype="bool")

        # Transform
        lrs = torch.stack([self.lr_transform(lr) for lr in lrs])
        qms = torch.stack([torch.from_numpy(qm) for qm in qms])
        hr = self.hr_transform(hr)
        sm = torch.from_numpy(sm)

        return dict(lr=lrs, qm=qms, hr=hr, sm=sm)
