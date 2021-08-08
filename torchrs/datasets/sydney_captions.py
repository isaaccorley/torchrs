import os

import torchvision.transforms as T

from torchrs.datasets import RSICD


class SydneyCaptions(RSICD):
    """ Sydney Captions dataset from 'Deep semantic understanding of
    high resolution remote sensing image', Qu et al (2016)
    https://ieeexplore.ieee.org/document/7546397

    'The Sydney dataset contains 7 different scene categories and totally has 613 HSR images
    ... Then every HSR image is annotated with 5 reference sentences
    ... has 613 images with 3065 captions'
    """
    splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = ".data/sydney_captions",
        split: str = "train",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        assert split in self.splits
        self.root = root
        self.transform = transform
        self.captions = self.load_captions(os.path.join(root, "dataset.json"), split)
        self.image_root = "images"
