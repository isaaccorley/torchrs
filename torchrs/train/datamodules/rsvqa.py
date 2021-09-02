from typing import Optional

from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import RSVQALR, RSVQAHR, RSVQAxBEN
from torchrs.transforms import Compose, ToTensor


class RSVQALRDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/RSVQA_LR",
        image_transform: Compose = Compose([ToTensor()]),
        text_transform: Compose = Compose([]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.image_transform = image_transform
        self.text_transform = text_transform

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = RSVQALR(root=self.root, split="train", image_transform=self.image_transform, text_transform=self.text_transform)
        self.val_dataset = RSVQALR(root=self.root, split="val", image_transform=self.image_transform, text_transform=self.text_transform)
        self.test_dataset = RSVQALR(root=self.root, split="test", image_transform=self.image_transform, text_transform=self.text_transform)


class RSVQAHRDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/RSVQA_HR",
        image_transform: Compose = Compose([ToTensor()]),
        text_transform: Compose = Compose([]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.image_transform = image_transform
        self.text_transform = text_transform

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = RSVQAHR(root=self.root, split="train", image_transform=self.image_transform, text_transform=self.text_transform)
        self.val_dataset = RSVQAHR(root=self.root, split="val", image_transform=self.image_transform, text_transform=self.text_transform)
        self.test_dataset = RSVQAHR(root=self.root, split="test", image_transform=self.image_transform, text_transform=self.text_transform)


class RSVQAxBENDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/rsvqaxben",
        image_transform: Compose = Compose([ToTensor()]),
        text_transform: Compose = Compose([]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.image_transform = image_transform
        self.text_transform = text_transform

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = RSVQAxBEN(root=self.root, split="train", image_transform=self.image_transform, text_transform=self.text_transform)
        self.val_dataset = RSVQAxBEN(root=self.root, split="val", image_transform=self.image_transform, text_transform=self.text_transform)
        self.test_dataset = RSVQAxBEN(root=self.root, split="test", image_transform=self.image_transform, text_transform=self.text_transform)
