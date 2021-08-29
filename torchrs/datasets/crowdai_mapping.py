import os
from glob import glob
from typing import List, Dict

import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from torchrs.transforms import Compose, ToTensor


class CrowdAIMapping(torch.utils.data.Dataset):
    """ The CrowdAI Mapping building instance segmentation dataset was proposed in
    'Deep Learning for Understanding Satellite Imagery: An Experimental Survey', Mohanty et al. (2020)
    https://www.frontiersin.org/articles/10.3389/frai.2020.534696/full

    The dataset was used in the CrowdAI Mapping Challenge
    https://www.aicrowd.com/challenges/mapping-challenge

    'The dataset used in this work was derived from the SpaceNet dataset.
    We only focus on the RGB channels.
    The dataset consists of a training set of 280,741 images, validation set of 60,317
    images and test set of 60,697 images.'
    """
    splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = ".data/CrowdAI-Mapping-Challenge",
        split: str = "train",
        transform: Compose = Compose([ToTensor()]),
    ):
        self.split = split
        self.transform = transform
        self.image_root = os.path.join(root, split, "images")
        self.images = glob(os.path.join(self.image_root, "*.jpg"))
        self.labels = self.load_labels(root, split)

    @staticmethod
    def load_labels(path: str, split: str) -> COCO:
        labels_path = os.path.join(path, split, "annotation.json")
        return COCO(labels_path)

    def __len__(self) -> int:
        return len(self.images)

    def load_masks(self, anns: List[Dict]) -> np.ndarray:
        return np.stack([self.labels.annToMask(ann) for ann in anns], axis=0)

    def load_boxes(self, masks: np.ndarray) -> np.ndarray:
        boxes = []
        for i in range(masks.shape[0]):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        return np.asarray(boxes)

    def __getitem__(self, idx: int) -> Dict:
        if self.split in ["train", "val"]:
            image = self.labels.imgs[idx]
            anns = self.labels.loadAnns(ids=[image["id"]])
            image_path = os.path.join(self.image_root, image["file_name"])
            x = np.array(Image.open(image_path).convert("RGB"))
            masks = self.load_masks(anns)
            boxes = self.load_boxes(masks)
            labels = np.ones((masks.shape[0]), dtype=np.uint8)
            iscrowd = np.array([ann["iscrowd"] for ann in anns], dtype=np.uint8)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            x, masks = self.transform([x, masks])
            output = dict(
                x=x, masks=masks, boxes=boxes, labels=labels,
                image_id=image["id"], iscrowd=iscrowd, area=area
            )
        else:
            image_path = self.images[idx]
            x = np.array(Image.open(image_path).convert("RGB"))
            self.transform(x)
            output = dict(x=x)

        return output
