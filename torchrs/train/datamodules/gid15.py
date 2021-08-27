from typing import Optional

from torchrs.transforms import Compose, ToTensor
from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import GID15


class GID15DataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/gid-15",
        transform: Compose = Compose([ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = GID15(root=self.root, split="train", transform=self.transform)
        self.val_dataset = GID15(root=self.root, split="val", transform=self.transform)
        self.test_dataset = GID15(root=self.root, split="test", transform=self.transform)
