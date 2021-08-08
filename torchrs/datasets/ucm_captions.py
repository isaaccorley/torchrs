import os

import torchvision.transforms as T

from torchrs.datasets import RSICD


class UCMCaptions(RSICD):
    """ UC Merced (UCM) Captions dataset from 'Deep semantic understanding of
    high resolution remote sensing image', Qu et al (2016)
    https://ieeexplore.ieee.org/document/7546397

    'The UCM dataset totally has 2100 HSR images which are divided into 21 challenging
    scene categories ... Then every HSR image is annotated with 5 reference sentences
    ... one has totally 2100 HSR remote sensing images with 10500 descriptions'
    """
    splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = ".data/ucm_captions",
        split: str = "train",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        assert split in self.splits
        self.root = root
        self.transform = transform
        self.captions = self.load_captions(os.path.join(root, "dataset.json"), split)
        self.image_root = "images"
