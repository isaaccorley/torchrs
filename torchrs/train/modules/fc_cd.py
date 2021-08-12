from typing import Dict

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

from torchrs.models import FCEF, FCSiamDiff, FCSiamConc


class BaseFCCDModule(pl.LightningModule):

    def __init__(
        self,
        channels: int = 3,
        t: int = 2,
        num_classes: int = 2,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        opt: torch.optim.Optimizer = torch.optim.Adam,
        lr: float = 3E-4
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.opt = opt
        self.lr = lr

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(threshold=0.5, num_classes=num_classes, average="micro", mdmc_average="global"),
            torchmetrics.Precision(num_classes=num_classes, threshold=0.5, average="micro", mdmc_average="global"),
            torchmetrics.Recall(num_classes=num_classes, threshold=0.5, average="micro", mdmc_average="global"),
            torchmetrics.F1(num_classes=num_classes, threshold=0.5, average="micro", mdmc_average="global"),
            torchmetrics.IoU(threshold=0.5, num_classes=num_classes),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def configure_optimizers(self):
        return self.opt(self.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        metrics = self.train_metrics(y_pred.softmax(dim=1), y)
        metrics["train_loss"] = loss
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        metrics = self.val_metrics(y_pred.softmax(dim=1), y)
        metrics["val_loss"] = loss
        self.log_dict(metrics)

    def test_step(self, batch: Dict, batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        metrics = self.test_metrics(y_pred.softmax(dim=1), y)
        metrics["test_loss"] = loss
        self.log_dict(metrics)


class FCEFModule(BaseFCCDModule):

    def __init__(
        self,
        channels: int = 3,
        t: int = 2,
        num_classes: int = 2,
        *args, **kwargs
    ):
        super().__init__(channels, t, num_classes, *args, **kwargs)
        self.model = FCEF(channels, t, num_classes)


class FCSiamConcModule(BaseFCCDModule):

    def __init__(
        self,
        channels: int = 3,
        t: int = 2,
        num_classes: int = 2,
        *args, **kwargs
    ):
        super().__init__(channels, t, num_classes, *args, **kwargs)
        self.model = FCSiamConc(channels, t, num_classes)


class FCSiamDiffModule(BaseFCCDModule):

    def __init__(
        self,
        channels: int = 3,
        t: int = 2,
        num_classes: int = 2,
        *args, **kwargs
    ):
        super().__init__(channels, t, num_classes, *args, **kwargs)
        self.model = FCSiamDiff(channels, t, num_classes)
