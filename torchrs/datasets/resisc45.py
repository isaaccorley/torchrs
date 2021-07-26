import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class RESISC45(ImageFolder):

    def __init__(
        self,
        root: str = ".data/NWPU-RESISC45",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__(
            root=root,
            transform=transform
        )
