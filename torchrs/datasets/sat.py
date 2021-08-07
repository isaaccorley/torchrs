import os
from typing import Tuple

import h5py
import torch
import torchvision.transforms as T

from torchrs.transforms import ToTensor


class SAT(torch.utils.data.Dataset):
    """ Base SAT dataset """
    def __init__(
        self,
        path: str = "",
        split: str = "train",
        transform: T.Compose = T.Compose([ToTensor()])
    ):
        assert split in ["train", "test"]
        self.path = path
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        with h5py.File(self.path, "r") as f:
            length = f[f"{self.split}_y"].shape[0]
        return length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.path, "r") as f:
            x = f[f"{self.split}_x"][idx]
            y = f[f"{self.split}_y"][idx]
        x = self.transform(x)
        y = torch.tensor(y).to(torch.long)
        return x, y


class SAT4(SAT):
    """ Base SAT dataset """
    classes = [
        "barren land",
        "trees",
        "grassland",
        "other"
    ]
    def __init__(
        self,
        path: str = ".data/sat/sat4.h5",
        split: str = "train",
        transform: T.Compose = T.Compose([ToTensor()])
    ):
        super().__init__(path, split, transform)


class SAT6(SAT4):
    classes = [
        "barren land",
        "trees",
        "grassland",
        "roads",
        "buildings",
        "water"
    ]
    def __init__(
        self,
        path: str = ".data/sat/sat6.h5",
        split: str = "train",
        transform: T.Compose = T.Compose([ToTensor()])
    ):
        super().__init__(path, split, transform)
