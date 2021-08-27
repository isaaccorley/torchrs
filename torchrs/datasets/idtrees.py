import os
from glob import glob
from xml.etree import ElementTree
from typing import List, Dict

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image


def parse_pascal_voc(path: str) -> Dict:
    et = ElementTree.parse(path)
    element = et.getroot()
    image = element.find("source").find("filename").text
    classes, points = [], []
    for obj in element.find("objects").findall("object"):
        obj_points = [p.text.split(",") for p in obj.find("points").findall("point")]
        obj_points = [(float(p1), float(p2)) for p1, p2 in obj_points]
        cls = obj.find("possibleresult").find("name").text
        classes.append(cls)
        points.append(obj_points)
    return dict(image=image, points=points, classes=classes)


class FAIR1M(torch.utils.data.Dataset):
    """ Integrating Data science with Trees and Remote Sensing (IDTReeS) dataset
    from the IDTReeS 2020 Competition
    https://idtrees.org/competition/

    """
    classes = {
        "Passenger Ship":   {"id": 0, "category": "Ship"},
        "Motorboat":        {"id": 1, "category": "Ship"},
    }

    def __init__(
        self,
        root: str = ".data/fair1m",
        transform: T.Compose = T.Compose([T.ToTensor()]),
    ):
        split = "train"
        self.root = root
        self.image_root = os.path.join(root, split, "part1", "images")
        self.transform = transform
        self.images = self.load_files(root, split)
        self.idx2cls = {i: c for i, c in enumerate(self.classes)}
        self.cls2idx = {c: i for i, c in self.idx2cls.items()}

    @staticmethod
    def load_files(root: str, split: str) -> List[Dict]:
        files = sorted(glob(os.path.join(root, split, "part1", "labelXmls", "*.xml")))
        return [parse_pascal_voc(f) for f in files]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, y, points where points is the x,y coords of the rotated bbox
        x: (3, h, w)
        y: (N,)
        points: (N, 5, 2)
        """
        image = self.images[idx]
        x = np.array(Image.open(os.path.join(self.image_root, image["image"])))
        x = x[..., :3]
        x = self.transform(x)
        y = torch.tensor([self.cls2idx[c] for c in image["classes"]])
        points = torch.tensor(image["points"])
        return dict(x=x, y=y, points=points)
