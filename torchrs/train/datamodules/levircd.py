from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchrs.datasets import LEVIRCDPlus
from torchrs.transforms import Compose, ToTensor


class LEVIRCDPlusDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root: str = ".data/levircd_plus",
        transform: Compose = Compose([ToTensor()]),
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = LEVIRCDPlus(root=self.root, split="train", transform=self.transform)
        self.test_dataset = LEVIRCDPlus(root=self.root, split="test", transform=self.transform)

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
