import torch
import torch.nn as nn


class SISRNet(nn.Module):

    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class RegNet(nn.Module):

    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class FusionNet(nn.Module):

    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepSUM(nn.Module):
    """Deep neural network for Super-resolution of Unregistered Multitemporal images
    https://arxiv.org/abs/1907.06490
    Parameters
    ----------
    scale_factor : int
        Super-Resolution scale factor. Determines Low-Resolution downsampling.
    channels: int
        Number of input and output channels
    num_blocks: int
        Number of stacked residual blocks
    """

    def __init__(
        self, scale_factor: int, channels: int = 3, num_blocks: int = 16
    ):
        super().__init__()
        self.sisrnet = SISRNet()
        self.regnet = RegNet()
        self.fusionnet = FusionNet()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve Low-Resolution input tensor
        Parameters
        ----------
        x : torch.Tensor
            Input Low-Resolution image as tensor
        Returns
        -------
        torch.Tensor
            Super-Resolved image as tensor
        """
        x = self.sisrnet(x)
        x = self.regnet(x)
        x = self.fusionnet(x)
        return x