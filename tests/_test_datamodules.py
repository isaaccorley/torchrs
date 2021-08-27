import pytest
import torch
import pytorch_lightning as pl

from torchrs.train import datamodules


skip = ["BaseDataModule", "RSVQAxBENDataModule", "S2MTCPDataModule"]


@torch.no_grad()
@pytest.mark.parametrize("datamodule", reversed(datamodules.__all__))
def test_datamodules(datamodule: pl.LightningDataModule):

    if datamodule in skip:
        return

    dm = getattr(datamodules, datamodule)()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    batch = next(iter(dm.val_dataloader()))
    batch = next(iter(dm.test_dataloader()))
