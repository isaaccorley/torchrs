from typing import Optional

from torchrs.transforms import Compose, ToTensor
from torchrs.datasets.utils import dataset_split
from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import LEVIRCDPlus


class LEVIRCDPlusDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/levircd_plus",
        transform: Compose = Compose([ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        train_dataset = LEVIRCDPlus(root=self.root, split="train", transform=self.transform)
        self.train_dataset, self.val_dataset = dataset_split(train_dataset, val_pct=self.val_split)
        self.test_dataset = LEVIRCDPlus(root=self.root, split="test", transform=self.transform)
