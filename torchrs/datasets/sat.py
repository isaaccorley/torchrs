import os

import h5py
import torchvision.transforms as T

from torchrs.transforms import ToTensor


class SAT4(torch.utils.data.Dataset):
    """ Image Captioning Dataset from 'Exploring Models and Data for
    Remote Sensing Image Caption Generation', Lu et al. (2017)
    https://arxiv.org/abs/1712.07835
    """
    classes = [
        "barren land",
        "trees",
        "grassland",
        "none"
    ]
    def __init__(
        self,
        root: str = ".data/sat",
        split: str = "train",
        transform: T.Compose = T.Compose([ToTensor()])
    ):
        assert split in ["train", "test"]
        self.path = os.path.join(root, "sat-4-full.mat")
        self.transform = transform

    def __len__(self) -> int:
        with h5py.File(self.path, "r") as f:
            length = f.keys()[f"{split}_x"].shape[0]
        return length

    def __getitem__(self, idx: int) -> Dict:
        captions = self.captions[idx]
        x = self.transform(x)
        sentences = [sentence["raw"] for sentence in captions["sentences"]]
        return dict(x=x, captions=sentences)
