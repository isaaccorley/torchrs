from typing import Optional

import torchvision.transforms as T

from torchrs.datasets.utils import dataset_split
from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import SAT4, SAT6


class SAT4DataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/sat/sat4.h5",
        transform: T.Compose = T.Compose([T.ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        train_dataset = SAT4(root=self.root, split="train", transform=self.transform)
        self.train_dataset, self.val_dataset = dataset_split(train_dataset, val_pct=self.val_split)
        self.test_dataset = SAT4(root=self.root, split="test", transform=self.transform)


class SAT6DataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/sat/sat6.h5",
        transform: T.Compose = T.Compose([T.ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        train_dataset = SAT6(root=self.root, split="train", transform=self.transform)
        self.train_dataset, self.val_dataset = dataset_split(train_dataset, val_pct=self.val_split)
        self.test_dataset = SAT6(root=self.root, split="test", transform=self.transform)
