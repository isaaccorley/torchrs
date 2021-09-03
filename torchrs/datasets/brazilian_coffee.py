import os
from glob import glob
from typing import Tuple, List

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image


class BrazilianCoffeeScenes(torch.utils.data.Dataset):
    """ Brazilian Coffee Scenes dataset from 'Do Deep Features Generalize from
    Everyday Objects to Remote Sensing and Aerial Scenes Domains?', Penatti at al. (2015)
    https://arxiv.org/abs/1703.00121
    """
    bands = ["Green", "Red", "NIR"]
    classes = ["non-coffee", "coffee"]

    def __init__(
        self,
        root: str = ".data/brazilian_coffee_scenes",
        transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__()
        self.transform = transform
        self.images, self.labels = self.load_images(root)

    @staticmethod
    def load_images(path: str) -> Tuple[List[str], List[int]]:
        folds = glob(os.path.join(path, "*.txt"))
        images, labels = [], []
        for fold in folds:
            fold_dir = os.path.join(path, os.path.splitext(os.path.basename(fold))[0])
            with open(fold, "r") as f:
                lines = f.read().strip().splitlines()

            for line in lines:
                label, image = line.split(".", 1)
                images.append(os.path.join(fold_dir, image + ".jpg"))
                labels.append(0 if label == "noncoffee" else 1)

        return images, labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, y = self.images[idx], self.labels[idx]
        x = np.array(Image.open(image).convert("RGB"))
        x = self.transform(x)
        return x, y
