from typing import Union, Any, Sequence, Callable

import torch
import numpy as np
import torchvision.transforms as T


__all__ = ["Compose", "ToTensor", "ToDtype"]


class Compose(T.Compose):

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, x: Union[Any, Sequence]):
        if isinstance(x, Sequence):
            for t in self.transforms:
                x = [t(i) for i in x]
        else:
            for t in self.transforms:
                x = t(x)
        return x


class ToTensor(object):

    def __init__(self, permute_dims: bool = True):
        self.permute_dims = permute_dims

    def __call__(self, x: np.ndarray) -> torch.Tensor:

        if x.dtype == "uint16":
            x = x.astype("int32")

        x = torch.from_numpy(x)

        if x.ndim == 2:
            if self.permute_dims:
                x = x[:, :, None]
            else:
                x = x[None, :, :]

        # Convert HWC->CHW
        if self.permute_dims:
            x = x.permute((2, 0, 1)).contiguous()

        return x


class ToDtype(object):

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.dtype)
