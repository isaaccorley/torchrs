import os
from glob import glob
from xml.etree import ElementTree
from typing import List, Dict

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image


class IDTrees(torch.utils.data.Dataset):
    """ Integrating Data science with Trees and Remote Sensing (IDTReeS) dataset
    from the IDTReeS 2020 Competition
    https://idtrees.org/competition/

    """
    classes = {
        "ACPE":     {"name": "Acer pensylvanicum L."},
        "ACRU":     {"name": "Acer rubrum L."},
        "ACSA3":    {"name": "Acer saccharum Marshall"},
        "AMLA":     {"name": "Amelanchier laevis Wiegand"},
        "BETUL":    {"name": "Betula sp."},
        "CAGL8":    {"name": "Carya glabra (Mill.) Sweet"},
        "CATO6":    {"name": "Carya tomentosa (Lam.) Nutt."},
        "FAGR":     {"name": "Fagus grandifolia Ehrh."},
        "GOLA":     {"name": "Gordonia lasianthus (L.) Ellis"},
        "LITU":     {"name": "Liriodendron tulipifera L."},
        "LYLU3":    {"name": "Lyonia lucida (Lam.) K. Koch"},
        "MAGNO":    {"name": "Magnolia sp."},
        "NYBI":     {"name": "Nyssa biflora Walter"},
        "NYSY":     {"name": "Nyssa sylvatica Marshall"},
        "OXYDE":    {"name": "Oxydendrum sp."},
        "PEPA37":   {"name": "Persea palustris (Raf.) Sarg."},
        "PIEL":     {"name": "Pinus elliottii Engelm."},
        "PIPA2":    {"name": "Pinus palustris Mill."},
        "PINUS":    {"name": "Pinus sp."},
        "PITA":     {"name": "Pinus taeda L."},
        "PRSE2":    {"name": "Prunus serotina Ehrh."},
        "QUAL":     {"name": "Quercus alba L."},
        "QUCO2":    {"name": "Quercus coccinea"},
        "QUGE2":    {"name": "Quercus geminata Small"},
        "QUHE2":    {"name": "Quercus hemisphaerica W. Bartram ex Willd."},
        "QULA2":    {"name": "Quercus laevis Walter"},
        "QULA3":    {"name": "Quercus laurifolia Michx."},
        "QUMO4":    {"name": "Quercus montana Willd."},
        "QUNI":     {"name": "Quercus nigra L."},
        "QURU":     {"name": "Quercus rubra L."},
        "QUERC":    {"name": "Quercus sp."},
        "ROPS":     {"name": "Robinia pseudoacacia L."},
        "TSCA":     {"name": "Tsuga canadensis (L.) Carriere"}
    }
    
    def __init__(
        self,
        root: str = ".data/idtrees",
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
