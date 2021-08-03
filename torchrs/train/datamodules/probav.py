from typing import Optional, Callable
from functools import partial

import torch
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchrs.datasets import PROBAV
from torchrs.datasets.probav import collate_fn
from torchrs.transforms import ToTensor, ToDtype


class PROBAVDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root: str = ".data/probav",
        band: str = "RED",
        lr_transform: T.Compose = T.Compose([ToTensor(), ToDtype(torch.float32)]),
        hr_transform: T.Compose = T.Compose([ToTensor(), ToDtype(torch.float32)]),
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        collate_fn: Optional[Callable] = collate_fn,
        test_collate_fn: Optional[Callable] = partial(collate_fn, shuffle=False)
    ):
        super().__init__()
        self.root = root
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.test_collate_fn = test_collate_fn

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
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            collate_fn=self.test_collate_fn
        )
