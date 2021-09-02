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


class RSVQA(torch.utils.data.Dataset):
    """ Base RSVQA dataset """
    splits = ["train", "val", "test"]
    prefix = ""
    image_root = ""

    def __init__(
        self,
        root: str = "",
        split: str = "train",
        image_transform: Compose = Compose([ToTensor()]),
        text_transform: Compose = Compose([])
    ):
        assert split in self.splits
        self.root = root
        self.split = split
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.image_root = os.path.join(root, self.image_root)
        self.ids, self.paths, self.images, self.questions, self.answers = self.load_files(
            self.root, self.image_root, self.split, self.prefix
        )

    @staticmethod
    def load_files(root: str, image_root: str, split: str, prefix: str) -> Tuple[List[int], List[str], List[Dict], List[Dict], List[Dict]]:
        paths = glob(os.path.join(image_root, "*.tif"))
        paths = sorted(paths, key=sort)
        with open(os.path.join(root, f"{prefix}_split_{split}_questions.json")) as f:
            questions = json.load(f)["questions"]
        with open(os.path.join(root, f"{prefix}_split_{split}_answers.json")) as f:
            answers = json.load(f)["answers"]
        with open(os.path.join(root, f"{prefix}_split_{split}_images.json")) as f:
            images = json.load(f)["images"]
        ids = [x["id"] for x in images if x["active"]]
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
        x = np.array(Image.open(os.path.join(self.image_root, f"{id}.tif")))
        x = self.image_transform(x)
        questions = [self.questions[i] for i in self.images[id]["questions_ids"]]
        answers = [self.answers[q["answers_ids"][0]]["answer"] for q in questions]
        types = [q["type"] for q in questions]
        questions = [q["question"] for q in questions]
        questions = self.text_transform(questions)
        answers = self.text_transform(answers)
        output = dict(x=x, questions=questions, answers=answers, types=types)
        return output


class RSVQALR(RSVQA):
    """Remote Sensing Visual Question Answering Low Resolution (RSVQA LR) dataset from
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
    image_root = "Images_LR"
    prefix = "LR"

    def __init__(self, root: str = ".data/RSVQA_LR", *args, **kwargs):
        super().__init__(root, *args, **kwargs)


class RSVQAHR(RSVQA):
    """Remote Sensing Visual Question Answering High Resolution (RSVQA HR) dataset from
    'RSVQA: Visual Question Answering for Remote Sensing Data', Lobry et al (2020)
    https://arxiv.org/abs/2003.07333

    'This dataset uses 15cm resolution aerial RGB images extracted from the High Resolution
    Orthoimagery (HRO) data collection of the USGS. This collection covers most urban areas of the
    USA, along with a few areas of interest (e.g. national parks). For most areas covered by the dataset,
    only one tile is available with acquisition dates ranging from year 2000 to 2016, with various sensors.
    The tiles are openly accessible through USGS' EarthExplorer tool.

    From this collection, we extracted 161 tiles belonging to the North-East coast of the USA
    that were split into 100659 images of size 512x512 (each covering 5898m^2).We constructed 100,660,316
    questions and answers following the methodology presented in subsection II-A. We split the data in
    a training set (61.5% of the tiles), a validation set (11.2%), and test sets (20.5% for test set 1,
    6.8% for test set 2). As it can be seen in Figure 4, test set 1 covers similar regions as the training
    and validation sets, while test set 2 covers the city of Philadelphia, which is not seen during the
    training. Note that this second test set also uses another sensor (marked as unknown on the USGS
    data catalog), not seen during training.
    """
    image_root = "Data"
    prefix = "USGS"

    def __init__(self, root: str = ".data/RSVQA_HR", *args, **kwargs):
        super().__init__(root, *args, **kwargs)


class RSVQAxBEN(RSVQA):
    """Remote Sensing Visual Question Answering BigEarthNet (RSVQAxBEN) dataset from
    'RSVQA Meets BigEarthNet: A New, Large-Scale, Visual Question Answering Dataset for Remote Sensing', Lobry et al (2021)
    https://rsvqa.sylvainlobry.com/IGARSS21.pdf

    'We introduce a new dataset to tackle the task of visual question answering on remote
    sensing images: this largescale, open access dataset extracts image/question/answer triplets
    from the BigEarthNet dataset. This new dataset contains close to 15 millions samples and is openly
    available.'
    """
    image_root = "Images"
    prefix = "RSVQAxBEN"

    def __init__(self, root: str = ".data/rsvqaxben", *args, **kwargs):
        super().__init__(root, *args, **kwargs)
