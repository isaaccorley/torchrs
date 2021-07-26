import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class RESISC45(ImageFolder):

    def __init__(
        self,
        root: str,
        transforms: T.Compose
    ):
        super().__init__(
            root=root,
            transform=transforms
        )
