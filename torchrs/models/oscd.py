import torch
import torch.nn as nn
from einops import rearrange


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.2, norm: bool = True):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


class EarlyFusion(nn.Module):
    """ Early Fusion (EF) from 'Urban Change Detection for Multispectral Earth Observation
    Using ConvolutionalNeural Networks', Daudt et al. (2018)
    https://arxiv.org/abs/1810.08468

    This model takes as input the concatenated image pairs (T*C, 15, 15)
    and is essentially a simple CNN classifier of the central pixel in an input patch.
    Assumes (T*Cx15x15) patch size input
    """
    def __init__(self, channels: int = 3, t: int = 2, num_classes: int = 2):
        super().__init__()
        filters = [channels * t, 32, 32, 64, 64, 128, 128]
        dropout = 0.2
        self.encoder = nn.Sequential(
            *[ConvBlock(filters[i-1], filters[i]) for i in range(1, len(filters))],
            ConvBlock(filters[-1], 128, dropout=0.0, norm=False),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Linear(8, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t c h w -> b (t c) h w")
        x = self.encoder(x)
        x = self.mlp(x)
        return x


class Siam(nn.Module):
    """ Siamese (Siam) from 'Urban Change Detection for Multispectral Earth Observation
    Using ConvolutionalNeural Networks', Daudt et al. (2018)
    https://arxiv.org/abs/1810.08468

    This model takes as input the concatenated image pairs (T*C, 15, 15)
    and is essentially a simple CNN classifier of the central pixel in an input patch.
    Assumes (T*Cx15x15) patch size input
    """
    def __init__(self, channels: int = 3, t: int = 2, num_classes: int = 2):
        super().__init__()
        filters = [channels, 64, 64, 128]
        dropout = 0.2
        self.encoder = nn.Sequential(
            *[ConvBlock(filters[i-1], filters[i]) for i in range(1, len(filters))],
            ConvBlock(filters[-1], 128, dropout=0.0),
        )
        self.mlp = nn.Sequential(
            nn.Linear(t*128*7*7, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.encoder(x)
        x = rearrange(x, "(b t) c h w -> b (t c h w)", b=b)
        x = self.mlp(x)
        return x
