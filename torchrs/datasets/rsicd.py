import os
import json
from typing import List, Dict

import torch
import torchvision.transforms as T
from PIL import Image


class RSICD(torch.utils.data.Dataset):
    """ Image Captioning Dataset from 'Exploring Models and Data for
    Remote Sensing Image Caption Generation', Lu et al. (2017)
    https://arxiv.org/abs/1712.07835

    'RSICD is used for remote sensing image captioning task. more than ten thousands
    remote sensing images are collected from Google Earth, Baidu Map, MapABC, Tianditu.
    The images are fixed to 224X224 pixels with various resolutions. The total number of
    remote sensing images are 10921, with five sentences descriptions per image.'
    """
    def __init__(
        self,
        root: str = ".data/rsicd",
        split: str = "train",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        assert split in ["train", "val", "test"]
        self.root = root
        self.transform = transform
        self.captions = self.load_captions(os.path.join(root, "dataset_rsicd.json"), split)

    @staticmethod
    def load_captions(path: str, split: str) -> List[Dict]:
        with open(path) as f:
            captions = json.load(f)["images"]
        return [c for c in captions if c["split"] == split]

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Dict:
        captions = self.captions[idx]
        path = os.path.join(self.root, "RSICD_images", captions["filename"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        sentences = [sentence["raw"] for sentence in captions["sentences"]]
        return dict(x=x, captions=sentences)
