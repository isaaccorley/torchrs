import pytest
import torch

from torchrs.transforms import ExtractChips


def test_extract_chips():
    x = torch.ones(3, 128, 128)
    f = ExtractChips((32, 32))
    z = f(x)
    assert z.shape == (16, 3, 32, 32)
