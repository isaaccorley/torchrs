import os
from glob import glob
from typing import Dict

import torch
import numpy as np
from PIL import Image

from torchrs.transforms import Compose, ToTensor


class LEVIRCDPlus(torch.utils.data.Dataset):
    """ LEVIR-CD+ dataset from 'S2Looking: A Satellite Side-Looking
    Dataset for Building Change Detection', Shen at al. (2021)
    https://arxiv.org/abs/2107.09244

    'LEVIR-CD+ contains more than 985 VHR (0.5m/pixel) bitemporal Google
    Earth images with dimensions of 1024x1024 pixels. These bitemporal images
    are from 20 different regions located in several cities in the state of
    Texas in the USA. The capture times of the image data vary from 2002 to
    2020. Images of different regions were taken at different times. The
    bitemporal images have a time span of 5 years.'
    """
    def __init__(
        self,
        root: str = ".data/levircd_plus",
        split: str = "train",
        transform: Compose = Compose([ToTensor()]),
    ):
        assert split in ["train", "test"]
        self.root = root
        self.transform = transform
        self.files = self.load_files(root, split)

    @staticmethod
    def load_files(root: str, split: str):
        files = []
        images = glob(os.path.join(root, split, "A", "*.png"))
        images = sorted([os.path.basename(image) for image in images])
        for image in images:
            image1 = os.path.join(root, split, "A", image)
            image2 = os.path.join(root, split, "B", image)
            mask = os.path.join(root, split, "label", image)
            files.append(dict(image1=image1, image2=image2, mask=mask))
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, mask
        x: (2, 13, h, w)
        mask: (1, h, w)
        """
        files = self.files[idx]
        mask = np.array(Image.open(files["mask"]))
        mask = np.clip(mask, 0, 1)
        image1 = np.array(Image.open(files["image1"]))
        image2 = np.array(Image.open(files["image2"]))
        image1, image2, mask = self.transform([image1, image2, mask])
        x = torch.stack([image1, image2], dim=0)
        return dict(x=x, mask=mask)
