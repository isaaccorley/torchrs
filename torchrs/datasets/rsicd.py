import os
import json
from typing import List, Dict, Optional

import torch
import torchvision.transforms as T
from PIL import Image

from torchrs.transforms import ToTensor


class RSICD(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str = ".data/rscid",
        annotations_path: Optional[str] = None,
        split: str = "train",
        transforms: T.Compose = T.Compose([ToTensor()])
    ):
        assert split in ["train", "val", "test"]
        self.root = root
        self.transforms = transforms

        
        self.annotations = self.load_annotations(annotations_path, split)
        print(f"RSICD {split} dataset loaded with {len(self.annotations)} annotations")

    def load_annotations(self, path: str, split: str) -> List[Dict]:
        with open(path) as f:
            annotations = json.load(f)["images"]

        return [a for a in annotations if a["split"] == split]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        annotation = self.annotations[idx]
        path = os.path.join(self.root, annotation["filename"])
        x = Image.open(path).convert("RGB")
        x = self.transforms(x)
        captions = [sentence["raw"] for sentence in annotation["sentences"]]
        return dict(x=x, captions=captions)
