import os
from glob import glob
from typing import Dict

import torch
import numpy as np
from PIL import Image

from torchrs.transforms import Compose, ToTensor


class S2Looking(torch.utils.data.Dataset):
    """ The Satellite Side-Looking (S2Looking) dataset from 'S2Looking: A Satellite Side-Looking
    Dataset for Building Change Detection', Shen at al. (2021)
    https://arxiv.org/abs/2107.09244

    'S2Looking is a building change detection dataset that contains large-scale side-looking
    satellite images captured at varying off-nadir angles. The S2Looking dataset consists of
    5,000 registered bitemporal image pairs (size of 1024*1024, 0.5 ~ 0.8 m/pixel) of rural
    areas throughout the world and more than 65,920 annotated change instances. We provide
    two label maps to separately indicate the newly built and demolished building regions
    for each sample in the dataset.'

    """
    def __init__(
        self,
        root: str = ".data/s2looking",
        split: str = "train",
        transform: Compose = Compose([ToTensor()]),
    ):
        assert split in ["train", "val", "test"]
        self.root = root
        self.transform = transform
        self.files = self.load_files(root, split)

    @staticmethod
    def load_files(root: str, split: str):
        files = []
        images = glob(os.path.join(root, split, "Image1", "*.png"))
        images = sorted([os.path.basename(image) for image in images])
        for image in images:
            image1 = os.path.join(root, split, "Image1", image)
            image2 = os.path.join(root, split, "Image2", image)
            build_mask = os.path.join(root, split, "label1", image)
            demo_mask = os.path.join(root, split, "label2", image)
            files.append(dict(image1=image1, image2=image2, build_mask=build_mask, demolish_mask=demo_mask))
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, mask
        x: (2, 13, h, w)
        build_mask: (1, h, w)
        demolish_mask: (1, h, w)
        """
        files = self.files[idx]
        build_mask = np.array(Image.open(files["build_mask"]))
        demo_mask = np.array(Image.open(files["demolish_mask"]))
        build_mask = np.clip(build_mask.mean(axis=-1), 0, 1).astype("uint8")
        demo_mask = np.clip(demo_mask.mean(axis=-1), 0, 1).astype("uint8")
        image1 = np.array(Image.open(files["image1"]))
        image2 = np.array(Image.open(files["image2"]))
        image1, image2, build_mask, demo_mask = self.transform([image1, image2, build_mask, demo_mask])
        x = torch.stack([image1, image2], dim=0)
        return dict(x=x, build_mask=build_mask, demolish_mask=demo_mask)
