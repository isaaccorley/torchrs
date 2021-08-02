import itertools
import pytest
import torch

from torchrs.models import RAMS


DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 32
T = [9, 11, 13, 15]
SCALE_FACTOR = [2, 3, 4]
CHANNELS = [1, 3]
BATCH_SIZE = [1, 2]

params = list(itertools.product(SCALE_FACTOR, T, CHANNELS, BATCH_SIZE))


@torch.no_grad()
@pytest.mark.parametrize("scale_factor, t, channels, batch_size", params)
def test_rams(scale_factor, t, channels, batch_size):
    model = RAMS(scale_factor, t, channels, num_feature_attn_blocks=3)
    model = model.to(DEVICE)
    model = model.eval()
    lr = torch.ones(batch_size, t, channels, IMAGE_SIZE, IMAGE_SIZE)
    lr = lr.to(DTYPE)
    lr = lr.to(DEVICE)
    sr = model(lr)
    assert sr.shape == (batch_size, channels, IMAGE_SIZE*scale_factor, IMAGE_SIZE*scale_factor)
    assert sr.dtype == torch.float32