import os
from glob import glob
from typing import List, Dict

import torch
import numpy as np

from torchrs.transforms import Compose, ToTensor


class S2MTCP(torch.utils.data.Dataset):
    """ Sentinel-2 Multitemporal Cities Pairs (S2MTCP) dataset from 'Self-supervised
    pre-training enhances change detection in Sentinel-2 imagery', Leenstra at al. (2021)
    https://arxiv.org/abs/2101.08122

    'A dataset of S2 level 1C image pairs was created ... Image pairs are selected randomly
    from available S2 images of each location with less than one percent cloud cover. Bands
    with a spatial resolution smaller than 10 m are resampled to 10 m and images are cropped
    to approximately 600x600 pixels centered on the selected coordinates ... The S2MTCP dataset
    contains N = 1520 image pairs, spread over all inhabited continents, with the highest
    concentration of image pairs in North-America, Europe and Asia'

    Note this dataset doesn't contain change masks as it was created for use in self-supervised pretraining
    """
    def __init__(
        self,
        root: str = ".data/s2mtcp",
        transform: Compose = Compose([ToTensor()]),
    ):
        self.root = os.path.join(root, "data_S21C")
        self.transform = transform
        self.files = self.load_files(self.root)

    @staticmethod
    def load_files(root: str) -> List[Dict]:
        files = glob(os.path.join(root, "*.npy"))
        files = [os.path.basename(f).split("_")[0] for f in files]
        files = sorted(set(files), key=int)
        files = [dict(image1=f"{num}_a.npy", image2=f"{num}_b.npy") for num in files]
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Returns x: (2, 14, h, w) """
        files = self.files[idx]
        image1 = np.load(os.path.join(self.root, files["image1"]))
        image2 = np.load(os.path.join(self.root, files["image2"]))
        image1, image2 = self.transform([image1, image2])
        x = torch.stack([image1, image2], dim=0)
        return x
