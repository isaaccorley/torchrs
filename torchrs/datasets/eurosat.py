import os

import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class EuroSATRGB(ImageFolder):

    def __init__(
        self,
        root: str,
        transforms: T.Compose
    ):
        super().__init__(
            root=os.path.join(root, "2750"),
            transform=transforms
        )

class EuroSATMS(ImageFolder):

    def __init__(
        self,
        root: str,
        transforms: T.Compose
    ):
        super().__init__(
            root=os.path.join(root, "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"),
            transform=transforms
        )
