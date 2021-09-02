import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class WHURS19(ImageFolder):
    """ WHU-RS19 dataset from'Structural High-resolution Satellite Image Indexing', Xia at al. (2010)
    https://hal.archives-ouvertes.fr/file/index/docid/458685/filename/structural_satellite_indexing_XYDG.pdf

    """
    def __init__(
        self,
        root: str = ".data/WHU-RS19",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__(
            root=root,
            transform=transform
        )
