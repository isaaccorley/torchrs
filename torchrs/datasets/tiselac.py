import os
from typing import Tuple

import torch
import numpy as np
from einops import rearrange

from torchrs.transforms import Compose, ToTensor


class Tiselac(torch.utils.data.Dataset):
    """ TiSeLac dataset from the Time Series Land Cover Classification Challenge (2017)
    https://sites.google.com/site/dinoienco/tiselac-time-series-land-cover-classification-challenge

    'A MSTC Land Cover classification problem for data taken from the Reunion island.
    A case is a pixel. Measurements are taken over 23 time points (days), with
    10 dimensions: 7 surface reflectances (Ultra Blue, Blue, Green, Red, NIR, SWIR1 and SWIR2)
    plus 3 indices (NDVI, NDWI and BI). Class values relate to one of 9 land cover types class values.'
    """
    classes = [
        "Urban Areas",
        "Other built-up surfaces",
        "Forests",
        "Sparse Vegetation",
        "Rocks and bare soil",
        "Grassland",
        "Sugarcane crops",
        "Other crops",
        "Water"
    ]
    splits = ["train", "test"]

    def __init__(
        self,
        root: str = ".data/tiselac",
        split: str = "train",
        transform: Compose = Compose([ToTensor(permute_dims=False)])
    ):
        assert split in self.splits
        self.root = root
        self.transform = transform
        self.series, self.labels = self.load_file(root, split)

    @staticmethod
    def load_file(path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
        x = np.loadtxt(os.path.join(path, f"{split}.txt"), dtype=np.int16, delimiter=",")
        y = np.loadtxt(os.path.join(path, f"{split}_labels.txt"), dtype=np.uint8)
        x = rearrange(x, "n (t c) -> n t c", c=10)
        return x, y

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.series[idx], self.labels[idx] - 1
        x, y = self.transform(x).squeeze(dim=0), torch.tensor(y).to(torch.long)
        return x, y
