import os
from glob import glob
from typing import List, Dict

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

from torchrs.transforms import ToTensor


class PROBAV(torch.utils.data.Dataset):
    """Multi-image super resolution (MISR) dataset from the PROBA-V Super Resolution Competition
    https://kelvins.esa.int/proba-v-super-resolution/home/

    'We collected satellite data from the PROBA-V mission of the European Space Agency from 74 hand-selected
    regions around the globe at different points in time. The data is composed of radiometrically and geometrically
    corrected Top-Of-Atmosphere (TOA) reflectances for the RED and NIR spectral bands at 300m and 100m resolution
    in Plate Carrée projection. The 300m resolution data is delivered as 128x128 grey-scale pixel images, the 100m
    resolution data as 384x384 grey-scale pixel images. The bit-depth of the images is 14, but they are saved in a
    16-bit .png-format (which makes them look relatively dark if opened in typical image viewers).

    Each image comes with a quality map, indicating which pixels in the image are concealed
    (i.e. clouds, cloud shadows, ice, water, missing, etc) and which should be considered clear. For an image to be
    included in the dataset, at least 75% of its pixels have to be clear for 100m resolution images, and 60% for
    300m resolution images. Each data-point consists of exactly one 100m resolution image and several 300m resolution
    images from the same scene. In total, the dataset contains 1450 scenes, which are split into 1160 scenes for
    training and 290 scenes for testing. On average, each scene comes with 19 different low resolution images and
    always with at least 9. We expect you to submit a 384x384 image for each of the 290 test-scenes, for which we
    will not provide a high resolution image.'
    """
    def __init__(
        self,
        root: str = ".data/probav",
        split: str = "train",
        band: str = "RED",
        lr_transform: T.Compose = T.Compose([ToTensor()]),
        hr_transform: T.Compose = T.Compose([ToTensor()]),
    ):
        assert split in ["train", "test"]
        assert band in ["RED", "NIR"]
        self.split = split
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.imgsets = self.load_files(root, split, band)

    @staticmethod
    def load_files(root: str, split: str, band: str) -> List[Dict]:
        imgsets = []
        folders = sorted(glob(os.path.join(root, split, band, "imgset*")))
        for folder in folders:
            lr = sorted(glob(os.path.join(folder, "LR*.png")))
            qm = sorted(glob(os.path.join(folder, "QM*.png")))
            sm = glob(os.path.join(folder, "SM.png"))[0]

            if split == "train":
                hr = glob(os.path.join(folder, "HR.png"))[0]
            else:
                hr = None

            imgsets.append(dict(lr=lr, qm=qm, hr=hr, sm=sm))
        return imgsets

    def __len__(self) -> int:
        return len(self.imgsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """ Returns a dict containing lr, qm, hr, sm
        lr: (t, 1, h, w) low resolution images
        qm: (t, 1, h, w) low resolution image quality masks
        hr: (1, h, w) high resolution image
        sm: (1, h, w) high resolution image status mask

        Note:
        lr/qm original size is (128, 128),
        hr/sm original size is (384, 384) (scale factor = 3)
        t is the number of lr images for an image set (min = 9)
        """
        imgset = self.imgsets[idx]

        # Load
        lrs = [np.array(Image.open(lr), dtype="int32") for lr in imgset["lr"]]
        qms = [np.array(Image.open(qm), dtype="bool") for qm in imgset["qm"]]
        sm = np.array(Image.open(imgset["sm"]), dtype="bool")

        # Transform
        lrs = torch.stack([self.lr_transform(lr) for lr in lrs])
        qms = torch.stack([torch.from_numpy(qm) for qm in qms]).unsqueeze(1)
        sm = torch.from_numpy(sm).unsqueeze(0)

        if self.split == "train":
            hr = np.array(Image.open(imgset["hr"]), dtype="int32")
            hr = self.hr_transform(hr)
            output = dict(lr=lrs, qm=qms, hr=hr, sm=sm)
        else:
            output = dict(lr=lrs, qm=qms, sm=sm)

        return output
