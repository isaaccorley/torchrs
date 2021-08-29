import os
from glob import glob
from typing import List, Tuple, Dict

import torch
import numpy as np
from PIL import Image

from torchrs.transforms import Compose, ToTensor


class DubaiSegmentation(torch.utils.data.Dataset):
    """ Semantic segmentation dataset of Dubai imagery taken by MBRSC satellites
    https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset/

    """
    classes = {
        "Unlabeled":            {"rgb": (155, 155, 155), "color": "#9B9B9B"},
        "Water":                {"rgb": (226, 169, 41),  "color": "#E2A929"},
        "Land (unpaved area)":  {"rgb": (132, 41, 246),  "color": "#8429F6"},
        "Road":                 {"rgb": (110, 193, 228), "color": "#6EC1E4"},
        "Building":             {"rgb": (60, 16, 152),   "color": "#3C1098"},
        "Vegetation":           {"rgb": (254, 221, 58),  "color": "#FEDD3A"}
    }
    colors = [v["rgb"] for k, v in classes.items()]

    def __init__(
        self,
        root: str = ".data/dubai-segmentation",
        transform: Compose = Compose([ToTensor()]),
    ):
        self.transform = transform
        self.images = self.load_images(root)
        self.regions = list(set([image["region"] for image in self.images]))

    @staticmethod
    def load_images(path: str) -> List[Dict]:
        images = sorted(glob(os.path.join(path, "**", "images", "*.jpg"), recursive=True))
        masks = sorted(glob(os.path.join(path, "**", "masks", "*.png"), recursive=True))
        regions = [image.split(os.sep)[-3] for image in images]
        files = [
            dict(image=image, mask=mask, region=region)
            for image, mask, region in zip(images, masks, regions)
        ]
        return files

    @staticmethod
    def rgb_to_mask(rgb: np.ndarray, colors: List[Tuple[int, int, int]]) -> np.ndarray:
        h, w = rgb.shape[:2]
        mask = np.zeros(shape=(h, w), dtype=np.uint8)
        for i, c in enumerate(colors):
            cmask = (rgb == c)
            if isinstance(cmask, np.ndarray):
                mask[cmask.all(axis=-1)] = i

        return mask

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        image_path, target_path = self.images[idx]["image"], self.images[idx]["mask"]
        x = np.array(Image.open(image_path).convert("RGB"))
        y = np.array(Image.open(target_path).convert("RGB"))
        y = self.rgb_to_mask(y, self.colors)
        x, y = self.transform([x, y])
        return dict(x=x, mask=y, region=self.images[idx]["region"])
