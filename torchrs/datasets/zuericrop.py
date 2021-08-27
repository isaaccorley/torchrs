import os
from typing import Dict

import h5py
import torch

from torchrs.transforms import Compose, ToTensor


class ZueriCrop(torch.utils.data.Dataset):
    """ ZueriCrop dataset from 'Crop mapping from image time series:
    deep learning with multi-scale label hierarchies', Turkoglu et al. (2021)
    https://arxiv.org/abs/2102.08820

    """
    classes = []

    def __init__(
        self,
        root: str = ".data/zuericrop",
        transform: Compose = Compose([ToTensor()]),
    ):
        self.transform = transform
        self.f = h5py.File(os.path.join(root, "ZueriCrop.hdf5"), "r")

    def __len__(self) -> int:
        return self.f["data"].shape[0]

    def __getitem__(self, idx: int) -> Dict:
        x = self.f["data"][idx, ...]
        mask = self.f["gt"][idx, ...]
        instance_mask = self.f["gt_instance"][idx, ...]
        x, mask, instance_mask = self.transform([x, mask, instance_mask])
        return dict(x=x, mask=mask, instance_mask=instance_mask)
