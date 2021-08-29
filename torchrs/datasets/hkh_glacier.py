import os
from glob import glob
from typing import List, Dict

import torch
import numpy as np

from torchrs.transforms import Compose, ToTensor


class HKHGlacierMapping(torch.utils.data.Dataset):
    """ Hindu Kush Himalayas (HKH) Glacier Mapping dataset
    https://lila.science/datasets/hkh-glacier-mapping

    'We also provide 14190 numpy patches. The numpy patches are all of size 512x512x15 and
    corresponding 512x512x2 pixel-wise mask labels; the two channels in the pixel-wise masks
    correspond to clean-iced and debris-covered glaciers. Patches' geolocation information,
    time stamps, source Landsat IDs, and glacier density are available in a geojson metadata file.'

    """
    bands = [
        "LE7 B1 (blue)",
        "LE7 B2 (green)",
        "LE7 B3 (red)",
        "LE7 B4 (near infrared)",
        "LE7 B5 (shortwave infrared 1)",
        "LE7 B6_VCID_1 (low-gain thermal infrared)",
        "LE7 B6_VCID_2 (high-gain thermal infrared)",
        "LE7 B7 (shortwave infrared 2)",
        "LE7 B8 (panchromatic)",
        "LE7 BQA (quality bitmask)",
        "NDVI (vegetation index)",
        "NDSI (snow index)",
        "NDWI (water index)",
        "SRTM 90 elevation",
        "SRTM 90 slope"
    ]

    def __init__(
        self,
        root: str = ".data/hkh_glacier_mapping",
        transform: Compose = Compose([ToTensor()]),
    ):
        self.transform = transform
        self.images = self.load_images(root)

    @staticmethod
    def load_images(path: str) -> List[Dict]:
        images = sorted(glob(os.path.join(path, "images", "*.npy")))
        masks = sorted(glob(os.path.join(path, "masks", "*.npy")))
        return [dict(image=image, mask=mask) for image, mask in zip(images, masks)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, target_path = self.images[idx]["image"], self.images[idx]["mask"]
        x, y = np.load(image_path), np.load(target_path)
        y0, y1 = y[..., 0], y[..., 1]
        x, y0, y1 = self.transform([x, y0, y1])
        return dict(x=x, clean_ice_mask=y0, debris_covered_mask=y1)
