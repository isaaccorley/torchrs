from typing import Optional

import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchrs.datasets import PROBAV
from torchrs.transforms import ToTensor


class PROBAVDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root: str = ".data/probav",
        band: str = "RED",
        lr_transform: T.Compose = T.Compose([ToTensor()]),
        hr_transform: T.Compose = T.Compose([ToTensor()]),
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False
    ):
        super().__init__()
        self.root = root
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = PROBAV(
            root=self.root, split="train", lr_transform=self.lr_transform, hr_transform=self.hr_transform
        )
        self.test_dataset = PROBAV(
            root=self.root, split="test", lr_transform=self.lr_transform, hr_transform=self.hr_transform
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory
        )
