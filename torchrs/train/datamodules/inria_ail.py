from typing import Optional

from torchrs.transforms import Compose, ToTensor
from torchrs.datasets.utils import dataset_split
from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import InriaAIL


class InriaAILDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/AerialImageDataset",
        transform: Compose = Compose([ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        train_dataset = InriaAIL(root=self.root, split="train", transform=self.transform)
        self.train_dataset, self.val_dataset = dataset_split(train_dataset, val_pct=self.val_split)
        self.test_dataset = InriaAIL(root=self.root, split="test", transform=self.transform)
