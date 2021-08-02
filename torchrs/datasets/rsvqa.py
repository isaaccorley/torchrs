import os
import json
from glob import glob
from typing import List, Dict, Tuple

import torch
import numpy as np
from PIL import Image

from torchrs.transforms import Compose, ToTensor


def sort(x):
    x = os.path.basename(x)
    x = os.path.splitext(x)[0]
    return int(x)


class RSVQALR(torch.utils.data.Dataset):
    """Remote Sensing Visual Question Answering (RSVQA) dataset from
    'RSVQA: Visual Question Answering for Remote Sensing Data', Lobry et al (2020)
    https://arxiv.org/abs/2003.07333

    'This dataset is based on Sentinel-2 images acquired over the Netherlands. Sentinel-2 satellites provide 10m resolution
    (for the visible bands used in this dataset) images with frequent updates (around 5 days) at a global scale. These images
    are openly available through ESA’s Copernicus Open Access Hub. To generate the dataset, we selected 9 Sentinel-2 tiles
    covering the Netherlands with a low cloud cover (selected tiles are shown in Figure 3). These tiles were divided in 772
    images of size 256x256 (covering 6.55km^2) retaining the RGB bands. From these, we constructed 770,232 questions and
    answers following the methodology presented in subsection II-A. We split the data in a training set (77.8% of the original tiles),
    a validation set (11.1%) and a test set (11.1%) at the tile level (the spatial split is shown in Figure 3). This allows
    to limit spatial correlation between the different splits.'
    """
    def __init__(
        self,
        root: str = ".data/RSVQA_LR",
        split: str = "train",
        transform: Compose = Compose([ToTensor()]),
    ):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.transform = transform
        self.ids, self.paths, self.images, self.questions, self.answers = self.load_files(root, split)

    @staticmethod
    def load_files(root: str, split: str) -> Tuple[List[int], List[str], List[Dict], List[Dict], List[Dict]]:
        paths = glob(os.path.join(root, "Images_LR", "*.tif"))
        paths = sorted(paths, key=sort)
        with open(os.path.join(root, "questions.json")) as f:
            questions = json.load(f)["questions"]
        with open(os.path.join(root, "answers.json")) as f:
            answers = json.load(f)["answers"]
        with open(os.path.join(root, "LR_split_train_images.json")) as f:
            train_images = json.load(f)["images"]
        with open(os.path.join(root, "LR_split_val_images.json")) as f:
            val_images = json.load(f)["images"]

        train_ids = [x["id"] for x in train_images if x["active"]]
        val_ids = [x["id"] for x in val_images if x["active"]]
        all_ids = [x["id"] for x in train_images]
        test_ids = list(set(all_ids) - set(train_ids + val_ids))

        if split == "train":
            ids = train_ids
            images = train_images
        elif split == "val":
            ids = val_ids
            images = val_images
        else:
            ids = test_ids
            images = []

        return ids, paths, images, questions, answers

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, questions, answers, q/a category
        x: (3, h, w)
        questions: List[str]
        answers: List[str]
        types: List[str]
        """
        id = self.ids[idx]
        x = np.array(Image.open(os.path.join(self.root, "Images_LR", f"{id}.tif")))
        x = self.transform(x)

        if self.split == "test":
            output = dict(x=x)
        else:
            questions = [self.questions[i] for i in self.images[id]["questions_ids"]]
            answers = [self.answers[q["answers_ids"][0]]["answer"] for q in questions]
            types = [q["type"] for q in questions]
            questions = [q["question"] for q in questions]
            output = dict(x=x, questions=questions, answers=answers, types=types)

        return output


class RSVQAxBEN(torch.utils.data.Dataset):
    """Remote Sensing Visual Question Answering BigEarthNet (RSVQAxBEN) dataset from
    'RSVQA Meets BigEarthNet: A New, Large-Scale, Visual Question Answering Dataset for Remote Sensing', Lobry et al (2021)
    https://rsvqa.sylvainlobry.com/IGARSS21.pdf

    'We introduce a new dataset to tackle the task of visual question answering on remote
    sensing images: this largescale, open access dataset extracts image/question/answer triplets
    from the BigEarthNet dataset. This new dataset contains close to 15 millions samples and is openly
    available.'
    """
    def __init__(
        self,
        root: str = ".data/rsvqaxben",
        split: str = "train",
        transform: Compose = Compose([ToTensor()]),
    ):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.transform = transform
        self.ids, self.paths, self.images, self.questions, self.answers = self.load_files(root, split)

    @staticmethod
    def load_files(root: str, split: str) -> Tuple[List[int], List[str], List[Dict], List[Dict], List[Dict]]:
        paths = glob(os.path.join(root, "Images", "*.tif"))
        paths = sorted(paths, key=sort)
        with open(os.path.join(root, f"RSVQAxBEN_split_{split}_questions.json")) as f:
            questions = json.load(f)["questions"]
        with open(os.path.join(root, f"RSVQAxBEN_split_{split}_answers.json")) as f:
            answers = json.load(f)["answers"]
        with open(os.path.join(root, f"RSVQAxBEN_split_{split}_images.json")) as f:
            images = json.load(f)["images"]
        ids = [x["id"] for x in images if x["active"]]
        return ids, paths, images, questions, answers

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing x, questions, answers, q/a category
        x: (3, h, w)
        questions: List[str]
        answers: List[str]
        types: List[str]
        """
        id = self.ids[idx]
        x = np.array(Image.open(os.path.join(self.root, "Images", f"{id}.tif")))
        x = self.transform(x)
        questions = [self.questions[i] for i in self.images[id]["questions_ids"]]
        answers = [self.answers[q["answers_ids"][0]]["answer"] for q in questions]
        types = [q["type"] for q in questions]
        questions = [q["question"] for q in questions]
        return dict(x=x, questions=questions, answers=answers, types=types)
