from typing import Optional

import torchvision.transforms as T

from torchrs.datasets.utils import dataset_split
from torchrs.train.datamodules import BaseDataModule
from torchrs.datasets import ADVANCE


class ADVANCEDataModule(BaseDataModule):

    def __init__(
        self,
        root: str = ".data/advance",
        image_transform: T.Compose = T.Compose([T.ToTensor()]),
        audio_transform: T.Compose = T.Compose([]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def setup(self, stage: Optional[str] = None):
        dataset = ADVANCE(
            root=self.root, image_transform=self.image_transform, audio_transform=self.audio_transform
        )
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            dataset, val_pct=self.val_split, test_pct=self.test_split
        )
