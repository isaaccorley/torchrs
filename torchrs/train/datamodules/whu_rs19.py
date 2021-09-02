from typing import Optional

import torchvision.transforms as T

from torchrs.datasets.utils import dataset_split
from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import WHU_RS19


class WHU_RS19DataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/WHU-RS19",
        transform: T.Compose = T.Compose([T.ToTensor()]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        dataset = WHU_RS19(root=self.root, transform=self.transform)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            dataset, val_pct=self.val_split, test_pct=self.test_split
        )
