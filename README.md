# PyTorch Remote Sensing (torchrs)

(WIP) PyTorch implementation of popular datasets and models in remote sensing tasks (Change Detection, Image Super Resolution, Land Cover Classification/Segmentation, Image-to-Image Translation, etc.) for various Optical (Sentinel-2, Landsat, etc.) and Synthetic Aperture Radar (SAR) (Sentinel-1) sensors.

## Installation

```bash
# pypi
pip install torch-rs

# latest
pip install git+https://github.com/isaaccorley/torchrs
```

## Table of Contents

* [Datasets](https://github.com/isaaccorley/torchrs#datasets)
* [Models](https://github.com/isaaccorley/torchrs#models)

## Datasets

* [PROBA-V Super Resolution](https://github.com/isaaccorley/torchrs#proba-v-super-resolution)
* [ETCI 2021 Flood Detection](https://github.com/isaaccorley/torchrs#etci-2021-flood-detection)
* [Remote Sensing Visual Question Answering (RSVQA) Low Resolution (LR)](https://github.com/isaaccorley/torchrs#remote-sensing-visual-question-answering-rsvqa-low-resolution-lr)
* [Remote Sensing Image Captioning Dataset (RSICD)](https://github.com/isaaccorley/torchrs#remote-sensing-image-captioning-dataset-rsicd)
* [Remote Sensing Image Scene Classification (RESISC45)](https://github.com/isaaccorley/torchrs#remote-sensing-image-scene-classification-resisc45)
* [EuroSAT](https://github.com/isaaccorley/torchrs#eurosat)

### PROBA-V Super Resolution

<img src="./assets/proba-v.jpg" width="500px"></img>

The [PROBA-V Super Resolution Challenge](https://kelvins.esa.int/proba-v-super-resolution/home/) dataset is a Multi-image Super Resolution (MISR) dataset of images taken by the [ESA PROBA-Vegetation satellite](https://earth.esa.int/eogateway/missions/proba-v). The dataset contains sets of unregistered 300m low resolution (LR) images which can be used to generate single 100m high resolution (HR) images for both Near Infrared (NIR) and Red bands. In addition, Quality Masks (QM) for each LR image and Status Masks (SM) for each HR image are available. The PROBA-V contains sensors which take imagery at 100m and 300m spatial resolutions with 5 and 1 day revisit rates, respectively. Generating high resolution imagery estimates would effectively increase the frequency at which HR imagery is available for vegetation monitoring.

The dataset can be downloaded using the `scripts/download_probav.sh` script and then used as below:

```python
from torchrs.transforms import Compose, ToTensor
from torchrs.datasets import PROBAV

transform = Compose([ToTensor()])

dataset = PROBAV(
    root="path/to/dataset/",
    split="train",          # or 'test'
    band="RED",             # or 'NIR'
    lr_transform=transform,
    hr_transform=transform
)

x = dataset[0]
"""
x: dict(
    lr: low res images  (t, 1, 128, 128)
    qm: quality masks   (t, 1, 128, 128)
    hr: high res image  (1, 384, 384)
    sm: status mask     (1, 384, 384)
)
t varies by set of images (minimum of 9)
"""
```

### ETCI 2021 Flood Detection

<img src="./assets/etci2021.jpg" width="450px"></img>

The [ETCI 2021 Dataset](https://nasa-impact.github.io/etci2021/) is a Flood Detection segmentation dataset of SAR images taken by the [ESA Sentinel-1 satellite](https://sentinel.esa.int/web/sentinel/missions/sentinel-1). The dataset contains pairs of VV and VH polarization images processed by the Hybrid Pluggable Processing Pipeline (hyp3) along with corresponding binary flood and water body ground truth masks.

The dataset can be downloaded using the `scripts/download_etci2021.sh` script and then used as below:

```python
from torchrs.transforms import Compose, ToTensor
from torchrs.datasets import ETCI2021

transform = Compose([ToTensor()])

dataset = ETCI2021(
    root="path/to/dataset/",
    split="train",          # or 'val', 'test'
    transform=transform
)

x = dataset[0]
"""
x: dict(
    vv:         (3, 256, 256)
    vh:         (3, 256, 256)
    flood_mask: (1, 256, 256)
    water_mask: (1, 256, 256)
)
"""
```

### Remote Sensing Visual Question Answering (RSVQA) Low Resolution (LR)

<img src="./assets/rsvqa_lr.png" width="800px"></img>

The [RSVQA LR](https://rsvqa.sylvainlobry.com/) dataset, proposed in ["RSVQA: Visual Question Answering for Remote Sensing Data", Lobry et al.](https://arxiv.org/abs/2003.07333) is a visual question answering (VQA) dataset of RGB images taken by the [ESA Sentinel-2 satellite](https://sentinel.esa.int/web/sentinel/missions/sentinel-2). Each image is annotated with a set of questions and their corresponding answers. Among other applications, this dataset can be used to train VQA models to perform scene understanding of medium resolution remote sensing imagery.

The dataset can be downloaded using the `scripts/download_rsvqa_lr.sh` script and then used as below:

```python
import torchvision.transforms as T
from torchrs.datasets import RSVQALR

transform = T.Compose([T.ToTensor()])

dataset = RSVQALR(
    root="path/to/dataset/",
    split="train",          # or 'val', 'test'
    transform=transform
)

x = dataset[0]
"""
x: dict(
    x:         (3, 256, 256)
    questions:  List[str]
    answers:    List[str]
    types:      List[str]
)
"""
```

### Remote Sensing Image Captioning Dataset (RSICD)

<img src="./assets/rsicd.png" width="500px"></img>

The [RSICD](https://github.com/201528014227051/RSICD_optimal) dataset, proposed in ["Exploring Models and Data for Remote Sensing Image Caption Generation", Lu et al.](https://arxiv.org/abs/1712.07835) is an image captioning dataset with 5 captions per image for 10,921 RGB images extracted using [Google Earth](https://earth.google.com/web/), [Baidu Map](https://map.baidu.com/), [MapABC](https://www.mapabc.com/) and [Tianditu](https://www.tianditu.gov.cn/). While one of the larger remote sensing image captioning datasets, this dataset contains very repetitive language with little detail and many captions are duplicated.

The dataset can be downloaded using the `scripts/download_rsicd.sh` script and then used as below:

```python
import torchvision.transforms as T
from torchrs.datasets import RSICD

transform = T.Compose([T.ToTensor()])

dataset = RSICD(
    root="path/to/dataset/",
    split="train",
    transform=transform
)

x = dataset[0]
"""
x: dict(
    x:        (3, 224, 224)
    captions: List[str]
)
"""
```

### Remote Sensing Image Scene Classification (RESISC45)

<img src="./assets/resisc45.png" width="500px"></img>

The [RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) dataset, proposed in ["Remote Sensing Image Scene Classification: Benchmark and State of the Art", Cheng et al.](https://arxiv.org/abs/1703.00121) is an image classification dataset of 31,500 RGB images extracted using [Google Earth Engine](https://earthengine.google.com/). The dataset contains 45 scenes with 700 images per class from over 100 countries and was selected to optimize for high variability in image conditions (spatial resolution, occlusion, weather, illumination, etc.).

The dataset can be downloaded using the `scripts/download_resisc45.sh` script and then used as below:

```python
import torchvision.transforms as T
from torchrs.datasets import RESISC45

transform = T.Compose([T.ToTensor()])

dataset = RESISC45(
    root="path/to/dataset/",
    transform=transform
)

x, y = dataset[0]
"""
x: (3, 256, 256)
y: int
"""

dataset.classes
"""
['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral',
'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest',
'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island',
'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot',
'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship',
'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']
"""
```

### EuroSAT

<img src="./assets/eurosat.jpg" width="600px"></img>

The [EuroSAT](https://github.com/phelber/eurosat) dataset, proposed in ["EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification", Helber et al.](https://arxiv.org/abs/1709.00029) is a land cover classification dataset of 27,000 images taken by the [ESA Sentinel-2 satellite](https://sentinel.esa.int/web/sentinel/missions/sentinel-2). The dataset contains 10 land cover classes with 2-3k images per class from over 34 European countries. The dataset is available in the form of RGB only or all [Multispectral (MS) Sentinel-2 bands](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial). This dataset is fairly easy with ~98.6% accuracy achieved with a ResNet-50.

The dataset can be downloaded using the `scripts/download_eurosat_rgb.sh` or `scripts/download_eurosat_ms.sh` scripts and then used as below:

```python
import torchvision.transforms as T
from torchrs.transforms import ToTensor
from torchrs.datasets import EuroSATRGB, EuroSATMS

transform = T.Compose([T.ToTensor()])

dataset = EuroSATRGB(
    root="path/to/dataset/",
    transform=transform
)

x, y = dataset[0]
"""
x: (3, 64, 64)
y: int
"""

transform = T.Compose([ToTensor()])

dataset = EuroSATMS(
    root="path/to/dataset/",
    transform=transform
)

x, y = dataset[0]
"""
x: (13, 64, 64)
y: int
"""

dataset.classes
"""
['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
"""
```

## Models

* [RAMS](https://github.com/isaaccorley/torchrs#rams)

### RAMS

<img src="./assets/rams.png" width="500px"></img>

Residual Attention Multi-image Super-resolution Network (RAMS) from 
["Multi-Image Super Resolution of Remotely Sensed Images Using Residual Attention Deep Neural Networks",
Salvetti et al. (2021)](https://www.mdpi.com/2072-4292/12/14/2207)

RAMS is currently one of the top performers on the [PROBA-V Super Resolution Challenge](https://kelvins.esa.int/proba-v-super-resolution/home/). This Multi-image Super Resolution (MISR) architecture utilizes attention based methods to extract spatial and spatiotemporal features from a set of unregistered low resolution images to form a single high resolution image.

```python
import torch
from torchrs.models import RAMS

# increase resolution by factor of 3 (e.g. 128x128 -> 384x384)
model = RAMS(
    scale_factor=3,
    t=9,
    c=1,
    num_feature_attn_blocks=12
)

# Input should be of shape (bs, t, c, h, w), where t is the number
# of low resolution input images and c is the number of channels/bands
lr = torch.randn(1, 9, 1, 128, 128)
sr = model(x) # (1, 1, 384, 384)
```

## Tests

```bash
$ pytest -ra
```
