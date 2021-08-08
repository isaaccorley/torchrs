from typing import Optional, Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        collate_fn: Optional[Callable] = None,
        test_collate_fn: Optional[Callable] = None,
        val_split: float = 0.1,
        test_split: float = 0.25
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.test_collate_fn = test_collate_fn
        self.val_split = val_split
        self.test_split = test_split

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

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            collate_fn=self.test_collate_fn
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
