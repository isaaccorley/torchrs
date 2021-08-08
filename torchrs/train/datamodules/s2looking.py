from typing import Optional

from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import S2Looking
from torchrs.transforms import Compose, ToTensor


class S2LookingDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/s2looking",
        transform: Compose = Compose([ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = S2Looking(root=self.root, split="train", transform=self.transform)
        self.val_dataset = S2Looking(root=self.root, split="val", transform=self.transform)
        self.test_dataset = S2Looking(root=self.root, split="test", transform=self.transform)
