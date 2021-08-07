import os
from glob import glob
from typing import List, Dict

import torch
import torchaudio
import numpy as np
import torchvision.transforms as T
from PIL import Image


class ADVANCE(torch.utils.data.Dataset):
    """ AuDio Visual Aerial sceNe reCognition datasEt (ADVANCE) from
    'Cross-Task Transfer for Geotagged Audiovisual Aerial Scene Recognition', Hu et al. (2020)
    https://arxiv.org/abs/2005.08449

    'We create an annotated dataset consisting of 5075 geotagged aerial imagesound pairs
    involving 13 scene classes. This dataset covers a large variety of scenes from across
    the world'
    """
    def __init__(
        self,
        root: str = ".data/advance",
        image_transform: T.Compose = T.Compose([T.ToTensor()]),
        audio_transform: T.Compose = T.Compose([]),
    ):
        self.root = root
        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.files = self.load_files(root)
        self.classes = sorted(set(f["cls"] for f in self.files))

    @staticmethod
    def load_files(root: str) -> List[Dict]:
        images = sorted(glob(os.path.join(root, "vision", "**", "*.jpg")))
        wavs = sorted(glob(os.path.join(root, "sound", "**", "*.wav")))
        labels = [image.split(os.sep)[-2] for image in images]
        files = [dict(image=image, audio=wav, cls=label) for image, wav, label in zip(images, wavs, labels)]
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """ Returns a dict containing image, audio, and class label
        image: (3, 512, 512)
        audio: (1, 220500)
        cls: int
        """
        files = self.files[idx]
        image = np.array(Image.open(files["image"]).convert("RGB"))
        audio, fs = torchaudio.load(files["audio"])
        image = self.image_transform(image)
        audio = self.audio_transform(audio)
        return dict(image=image, audio=audio, cls=files["cls"])
