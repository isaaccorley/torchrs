import torch
import numpy as np

class ToTensor(object):

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)
