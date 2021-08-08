from typing import Tuple

import h5py
import torch
import torchvision.transforms as T


class SAT(torch.utils.data.Dataset):
    """ Base SAT dataset """
    splits = ["train", "test"]

    def __init__(
        self,
        root: str = "",
        split: str = "train",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        assert split in self.splits
        self.root = root
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        with h5py.File(self.root, "r") as f:
            length = f[f"{self.split}_y"].shape[0]
        return length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.root, "r") as f:
            x = f[f"{self.split}_x"][idx]
            y = f[f"{self.split}_y"][idx]
        x = self.transform(x)
        return x, y


class SAT4(SAT):
    """ SAT-4 land cover classification dataset from "DeepSat - A Learning framework
    for Satellite Imagery", Basu et al (2015)
    https://arxiv.org/abs/1509.03602

    'SAT-4 consists of a total of 500,000 image patches covering four broad land cover classes.
    These include - barren land, trees, grassland and a class that consists of all land cover
    classes other than the above three. 400,000 patches (comprising of four-fifths of the total
    dataset) were chosen for training and the remaining 100,000 (one-fifths) were chosen as the testing
    dataset. We ensured that the training and test datasets belong to disjoint set of image tiles.
    Each image patch is size normalized to 28x28 pixels. Once generated, both the training and testing
    datasets were randomized using a pseudo-random number generator.'
    """
    classes = [
        "barren land",
        "trees",
        "grassland",
        "other"
    ]

    def __init__(self, root: str = ".data/sat/sat4.h5", *args, **kwargs):
        super().__init__(root, *args, **kwargs)


class SAT6(SAT):
    """ SAT-6 land cover classification dataset from "DeepSat - A Learning framework
    for Satellite Imagery", Basu et al (2015)
    https://arxiv.org/abs/1509.03602

    'SAT-6 consists of a total of 405,000 image patches each of size 28x28 and covering 6
    landcover classes - barren land, trees, grassland, roads, buildings and water bodies.
    324,000 images (comprising of four-fifths of the total dataset) were chosen as the training
    dataset and 81,000 (one fifths) were chosen as the testing dataset. Similar to SAT-4,
    the training and test sets were selected from disjoint NAIP tiles. Once generated, the
    images in the dataset were randomized in the same way as that for SAT-4. The specifications
    for the various landcover classes of SAT-4 and SAT-6 were adopted from those used in the
    National Land Cover Data (NLCD) algorithm.'
    """
    classes = [
        "barren land",
        "trees",
        "grassland",
        "roads",
        "buildings",
        "water"
    ]

    def __init__(self, root: str = ".data/sat/sat6.h5", *args, **kwargs):
        super().__init__(root, *args, **kwargs)
