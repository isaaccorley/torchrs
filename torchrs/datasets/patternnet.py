import os

import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class PatternNet(ImageFolder):
    """ PatternNet dataset from 'PatternNet: A benchmark dataset for performance
    evaluation of remote sensing image retrieval', Zhou at al. (2018)
    https://arxiv.org/abs/1706.03424

    """
    def __init__(
        self,
        root: str = ".data/PatternNet",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__(
            root=os.path.join(root, "images"),
            transform=transform
        )
