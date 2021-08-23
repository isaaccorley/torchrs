import os
import json
from glob import glob
from typing import List, Dict

import torch
import numpy as np

from torchrs.transforms import Compose, ToTensor


class PASTIS(torch.utils.data.Dataset):
    """ Image Captioning Dataset from 'Exploring Models and Data for
    Remote Sensing Image Caption Generation', Lu et al. (2017)
    https://arxiv.org/abs/1712.07835

    'PASTIS is a benchmark dataset for panoptic and semantic segmentation of
    agricultural parcels from satellite time series. It contains 2,433 patches
    within the French metropolitan territory with panoptic annotations
    (instance index + semantic label for each pixel). Each patch is a Sentinel-2
    multispectral image time series of variable length.'
    """
    classes = [
        "Background",
        "Meadow",
        "Soft winter wheat",
        "Corn",
        "Winter barley",
        "Winter rapeseed",
        "Spring barley",
        "Sunflower",
        "Grapevine",
        "Beet",
        "Winter triticale",
        "Winter durum wheat",
        "Fruits,  vegetables, flowers",
        "Potatoes",
        "Leguminous fodder",
        "Soybeans",
        "Orchard",
        "Mixed cereal",
        "Sorghum",
        "Void label"
    ]
    colormap = [
        (0, 0, 0),
        (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
        (1.0, 0.4980392156862745, 0.054901960784313725),
        (1.0, 0.7333333333333333, 0.47058823529411764),
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
        (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
        (1.0, 0.596078431372549, 0.5882352941176471),
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
        (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
        (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
        (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
        (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
        (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
        (1, 1, 1)
    ]

    def __init__(
        self,
        root: str = ".data/pastis",
        transform: Compose = Compose([ToTensor()]),
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.images = self.load_images(self.image_root)

    @staticmethod
    def load_images(path: str) -> List[Dict]:
        parcel_prefix, target_prefix = "ParcelIDs_", "TARGET_"
        files = glob(os.path.join(path, "ANNOTATIONS", f"{parcel_prefix}*.npy"))
        return [dict(image=f, target=f.replace(parcel_prefix, target_prefix)) for f in files]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        image_path, target_path = self.images[idx]["image"], self.images[idx]["target"]
        x, y = np.load(image_path), np.load(target_path)
        x, y = self.transform([x, y])
        return dict(x=x, mask=y)
