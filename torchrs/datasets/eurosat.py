import os

import tifffile
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from torchrs.transforms import ToTensor


class EuroSATRGB(ImageFolder):

    def __init__(
        self,
        root: str = ".data/eurosat-rgb",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__(
            root=os.path.join(root, "2750"),
            transform=transform
        )


class EuroSATMS(ImageFolder):

    def __init__(
        self,
        root: str = ".data/eurosat-ms",
        transform: T.Compose = T.Compose([ToTensor()])
    ):
        super().__init__(
            root=os.path.join(root, "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"),
            transform=transform,
            loader=tifffile.imread
        )
