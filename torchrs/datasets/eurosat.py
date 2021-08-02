import os

import tifffile
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from torchrs.transforms import ToTensor


class EuroSATRGB(ImageFolder):
    """ Sentinel-2 RGB Land Cover Classification dataset from 'EuroSAT: A Novel Dataset
    and Deep Learning Benchmark for Land Use and Land Cover Classification', Helber at al. (2017)
    https://arxiv.org/abs/1709.00029

    'We present a novel dataset based on Sentinel-2 satellite images covering 13 spectral
    bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images.'

    Note: RGB bands only
    """
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
    """ Sentinel-2 RGB Land Cover Classification dataset from 'EuroSAT: A Novel Dataset
    and Deep Learning Benchmark for Land Use and Land Cover Classification', Helber at al. (2017)
    https://arxiv.org/abs/1709.00029

    'We present a novel dataset based on Sentinel-2 satellite images covering 13 spectral
    bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images.'

    Note: all 13 multispectral (MS) bands
    """
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
