from typing import Optional

import torchvision.transforms as T

from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import RSICD


class RSICDDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/rsicd",
        transform: T.Compose = T.Compose([T.ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = RSICD(root=self.root, split="train", transform=self.transform)
        self.val_dataset = RSICD(root=self.root, split="val", transform=self.transform)
        self.test_dataset = RSICD(root=self.root, split="test", transform=self.transform)
