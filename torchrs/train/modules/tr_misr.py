from typing import Dict

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

from torchrs.models import TRMISR


class TRMISRModule(pl.LightningModule):

    def __init__(
        self,
        scale_factor: int = 3,
        input_dim: int = 32,
        channels: int = 1,
        t: int = 9,
        num_layers: int = 4,
        num_heads: int = 8,
        pool: str = "cls",
        loss_fn: nn.Module = nn.MSELoss(),
        opt: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 3E-4
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.opt = opt
        self.lr = lr
        self.model = TRMISR(
            scale_factor, input_dim, channels,
            t, num_layers, num_heads, pool
        )

        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanSquaredError(),
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanAbsolutePercentageError(),
            torchmetrics.PSNR(),
            torchmetrics.SSIM()
        ])
        self.train_metrics = metrics.clone(prefix='train_')
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
        metrics = self.train_metrics(sr.to(torch.float32), hr)
        metrics["train_loss"] = loss
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        lr, hr = batch["lr"], batch["hr"]
        sr = self(lr)
        loss = self.loss_fn(sr, hr)
        metrics = self.val_metrics(sr.to(torch.float32), hr)
        metrics["val_loss"] = loss
        self.log_dict(metrics)

    def test_step(self, batch: Dict, batch_idx: int):
        lr, hr = batch["lr"], batch["hr"]
        sr = self(lr)
        loss = self.loss_fn(sr, hr)
        metrics = self.test_metrics(sr.to(torch.float32), hr)
        metrics["test_loss"] = loss
        self.log_dict(metrics)
