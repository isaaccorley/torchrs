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
    """ FAIR1M dataset from 'FAIR1M: A Benchmark Dataset
    for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery', Sun at al. (2021)
    https://arxiv.org/abs/2103.05569

    'We propose a novel benchmark dataset with more than 1 million instances and more
    than 15,000 images for Fine-grained object recognition in high-resolution remote
    sensing imagery which is named as FAIR1M. We collected remote sensing images with
    a resolution of 0.3m to 0.8m from different platforms, which are spread across many
    countries and regions. All objects in the FAIR1M dataset are annotated with respect
    to 5 categories and 37 sub-categories by oriented bounding boxes.'
    """
    classes = {
        "Passenger Ship":   {"id": 0, "category": "Ship"},
        "Motorboat":        {"id": 1, "category": "Ship"},
        "Fishing Boat":     {"id": 2, "category": "Ship"},
        "Tugboat":          {"id": 3, "category": "Ship"},
        "other-ship":       {"id": 4, "category": "Ship"},
        "Engineering Ship": {"id": 5, "category": "Ship"},
        "Liquid Cargo Ship": {"id": 6, "category": "Ship"},
        "Dry Cargo Ship":   {"id": 7, "category": "Ship"},
        "Warship":          {"id": 8, "category": "Ship"},
        "Small Car":        {"id": 9, "category": "Vehicle"},
        "Bus":              {"id": 10, "category": "Vehicle"},
        "Cargo Truck":      {"id": 11, "category": "Vehicle"},
        "Dump Truck":       {"id": 12, "category": "Vehicle"},
        "other-vehicle":    {"id": 13, "category": "Vehicle"},
        "Van":              {"id": 14, "category": "Vehicle"},
        "Trailer":          {"id": 15, "category": "Vehicle"},
        "Tractor":          {"id": 16, "category": "Vehicle"},
        "Excavator":        {"id": 17, "category": "Vehicle"},
        "Truck Tractor":    {"id": 18, "category": "Vehicle"},
        "Boeing737":        {"id": 19, "category": "Airplane"},
        "Boeing747":        {"id": 20, "category": "Airplane"},
        "Boeing777":        {"id": 21, "category": "Airplane"},
        "Boeing787":        {"id": 22, "category": "Airplane"},
        "ARJ21":            {"id": 23, "category": "Airplane"},
        "C919":             {"id": 24, "category": "Airplane"},
        "A220":             {"id": 25, "category": "Airplane"},
        "A321":             {"id": 26, "category": "Airplane"},
        "A330":             {"id": 27, "category": "Airplane"},
        "A350":             {"id": 28, "category": "Airplane"},
        "other-airplane":   {"id": 29, "category": "Airplane"},
        "Baseball Field":   {"id": 30, "category": "Court"},
        "Basketball Court": {"id": 31, "category": "Court"},
        "Football Field":   {"id": 32, "category": "Court"},
        "Tennis Court":     {"id": 33, "category": "Court"},
        "Roundabout":       {"id": 34, "category": "Road"},
        "Intersection":     {"id": 35, "category": "Road"},
        "Bridge":           {"id": 36, "category": "Road"}
    }

    def __init__(
        self,
        root: str = ".data/fair1m",
        split: str = "train",
        transform: T.Compose = T.Compose([T.ToTensor()]),
    ):
        assert split in ["train"]
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
