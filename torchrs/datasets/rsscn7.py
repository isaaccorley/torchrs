import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class RSSCN7(ImageFolder):
    """ RSSCN7 dataset from 'Deep Learning Based Feature Selection for
    Remote Sensing Scene Classification', Zou at al. (2015)
    https://ieeexplore.ieee.org/abstract/document/7272047

    'The data set RSSCN7 contains 2800 remote sensing scene images, which
    are from seven typical scene categories, namely, the grassland, forest,
    farmland, parking lot, residential region, industrial region, and river
    and lake. For each category, there are 400 images collected from the
    Google Earth, which are sampled on four different scales with 100 images
    per scale. Each image has a size of 400x400 pixels. This data set is
    rather challenging due to the wide diversity of the scene images that
    are captured under changing seasons and varying weathers and sampled
    on different scales'
    """
    def __init__(
        self,
        root: str = ".data/RSSCN7",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__(
            root=root,
            transform=transform
        )
