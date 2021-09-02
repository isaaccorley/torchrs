import os

import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class UCM(ImageFolder):
    """ UC Merced Land Use dataset from 'Bag-Of-Visual-Words and
    Spatial Extensions for Land-Use Classification', Yang at al. (2010)
    https://faculty.ucmerced.edu/snewsam/papers/Yang_ACMGIS10_BagOfVisualWords.pdf

    """
    def __init__(
        self,
        root: str = ".data/UCMerced_LandUse",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__(
            root=os.path.join(root, "Images"),
            transform=transform
        )
