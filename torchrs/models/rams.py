import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


class TemporalAttention(nn.Module):

    def __init__(self, channels: int, kernel_size: int, r: int):
        super().__init__()
        self.model = nn.Sequential(
            Reduce("b c h w -> b c () ()", "mean"),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels//r,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels//r,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.model(x)


class FeatureAttention(nn.Module):

    def __init__(self, channels: int, kernel_size: int, r: int):
        super().__init__()
        self.model = nn.Sequential(
            Reduce("b t c h w -> b () c () ()", "mean"),
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels//r,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=channels//r,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)

class RTAB(nn.Module):
    """ Residual Temporal Attention Block """
    def __init__(self, channels: int, kernel_size: int, r: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            TemporalAttention(channels=channels, kernel_size=kernel_size, r=r)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class RFAB(nn.Module):
    """ Residual Feature Attention Block """
    def __init__(self, channels: int, kernel_size: int, r: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            FeatureAttention(channels=channels, kernel_size=kernel_size, r=r)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)



class RAMS(nn.Module):
    """
    https://www.mdpi.com/2072-4292/12/14/2207
    """
    def __init__(
        self,
        scale_factor: int = 2,
        t: int = 9,
        channels: int = 1,
        num_feature_attn_blocks: int = 12,
    ):
        super().__init__()
        filters = 32
        kernel_size = 3
        r = 8
        self.temporal_attn = nn.Sequential(
            Rearrange("b t c h w -> b (t c) h w"),
            RTAB(channels=t*c, kernel_size=kernel_size, r=r),
            nn.Conv2d(
                in_channels=t*c,
                out_channels=scale_factor * c,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size//2
            ),
            nn.PixelShuffle(scale_factor),
            Rearrange("b (t c) h w -> b t c h w", t=t, c=c)
        )
        self.head = nn.Sequential()
        self.feature_attn = nn.Sequential()
        self.temporal_redn = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        t_attn = self.temporal_attn(x)
        f_attn = x + self.feature_attn(x)
        return t_attn + 