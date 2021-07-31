from typing import Dict

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

from torchrs.models import RAMS


class RAMSModule(pl.LightningModule):

    def __init__(
        self,
        scale_factor: int = 3,
        t: int = 9,
        c: int = 1,
        num_feature_attn_blocks: int = 12,
        loss_fn: nn.Module = nn.MSELoss(),
        opt: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 3E-4
    ):
        super(RAMSModule, self).__init__()
        self.loss_fn = loss_fn
        self.opt = opt
        self.lr = lr
        self.model = RAMS(scale_factor, t, c, num_feature_attn_blocks)

        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanSquaredError(),
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanAbsolutePercentageError(),
            torchmetrics.PSNR(),
            torchmetrics.SSIM()
        ])
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def configure_optimizers(self):
        return self.opt(self.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int):
        lr, hr = batch["lr"], batch["hr"]
        sr = self(lr)
        loss = self.loss_fn(sr, hr)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        lr, hr = batch["lr"], batch["hr"]
        sr = self(lr)
        loss = self.loss_fn(sr, hr)
        metrics = self.val_metrics(sr, hr)
        metrics["val_loss"] = loss
        self.log_dict(metrics)

    def test_step(self, batch: Dict, batch_idx: int):
        lr, hr = batch["lr"], batch["hr"]
        sr = self(lr)
        loss = self.loss_fn(sr, hr)
        metrics = self.test_metrics(sr, hr)
        metrics["test_loss"] = loss
        self.log_dict(metrics)
