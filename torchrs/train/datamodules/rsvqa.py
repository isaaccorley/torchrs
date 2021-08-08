from typing import Optional

from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import RSVQALR, RSVQAxBEN
from torchrs.transforms import Compose, ToTensor


class RSVQALRDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/RSVQA_LR",
        transform: Compose = Compose([ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = RSVQALR(root=self.root, split="train", transform=self.transform)
        self.val_dataset = RSVQALR(root=self.root, split="val", transform=self.transform)
        self.test_dataset = RSVQALR(root=self.root, split="test", transform=self.transform)


class RSVQAxBENDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/rsvqaxben",
        transform: Compose = Compose([ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = RSVQAxBEN(root=self.root, split="train", transform=self.transform)
        self.val_dataset = RSVQAxBEN(root=self.root, split="val", transform=self.transform)
        self.test_dataset = RSVQAxBEN(root=self.root, split="test", transform=self.transform)
