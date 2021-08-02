""" Referenced from official TF implementation https://github.com/EscVM/RAMS/blob/master/utils/network.py """
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


class ReflectionPad3d(nn.Module):
    """ Custom 3D reflection padding for only h, w dims """
    def __init__(self, padding):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> b (t c) h w")
        x = self.pad(x)
        x = rearrange(x, "b (t c) h w -> b t c h w", t=t, c=c)
        return x


class TemporalAttention(nn.Module):
    """ Temporal Attention Block """
    def __init__(self, channels: int, kernel_size: int, r: int):
        super().__init__()
        self.model = nn.Sequential(
            Reduce("b c h w -> b c () ()", "mean"),
            nn.Conv2d(channels, channels//r, kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(channels//r, channels, kernel_size, stride=1, padding=kernel_size//2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.model(x)


class FeatureAttention(nn.Module):
    """ Feature Attention Block """
    def __init__(self, channels: int, kernel_size: int, r: int):
        super().__init__()
        self.model = nn.Sequential(
            Reduce("b c t h w -> b c () () ()", "mean"),
            nn.Conv3d(channels, channels//r, kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv3d(channels//r, channels, kernel_size, stride=1, padding=kernel_size//2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.model(x)


class RTAB(nn.Module):
    """ Residual Temporal Attention Block """
    def __init__(self, channels: int, kernel_size: int, r: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2),
            TemporalAttention(channels, kernel_size, r)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class RFAB(nn.Module):
    """ Residual Feature Attention Block """
    def __init__(self, channels: int, kernel_size: int, r: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2),
            FeatureAttention(channels, kernel_size, r)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class TemporalReductionBlock(nn.Module):
    """ Temporal Reduction Block """
    def __init__(self, channels: int, kernel_size: int, r: int):
        super().__init__()
        self.model = nn.Sequential(
            ReflectionPad3d(1),
            RFAB(channels, kernel_size, r),
            nn.Conv3d(channels, channels, kernel_size, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RAMS(nn.Module):
    """
    Residual Attention Multi-image Super-resolution Network (RAMS)
    'Multi-Image Super Resolution of Remotely Sensed Images Using Residual Attention Deep Neural Networks'
    Salvetti et al. (2021)
    https://www.mdpi.com/2072-4292/12/14/2207

    Note this model was built to work with t=9 input images and kernel_size=3. Other values may not work.
    t must satisfy the constraints of ((t-1)/(kernel_size-1) - 1) % 1 == 0 where kernel_size=3 and t >= 5.
    Some valid t's are [9, 11, 13, ...]
    """
    def __init__(
        self,
        scale_factor: int = 3,
        t: int = 9,
        c: int = 1,
        num_feature_attn_blocks: int = 12,
    ):
        super().__init__()
        filters = 32
        kernel_size = 3
        r = 8
        num_temporal_redn_blocks = ((t-1)/(kernel_size-1) - 1)
        err = """t must satisfy the ((t-1)/(kernel_size-1) - 1) % 1 == 0 where kernel_size=3
                and t >= 5. Some valid t's are [9, 11, 13, 15, ...] """
        assert num_temporal_redn_blocks % 1 == 0 and t >= 9, err

        self.temporal_attn = nn.Sequential(
            Rearrange("b t c h w -> b (t c) h w"),
            nn.ReflectionPad2d(1),
            RTAB(t * c, kernel_size, r),
        )
        self.residual_upsample = nn.Sequential(
            nn.Conv2d(t * c, c * scale_factor ** 2, kernel_size, stride=1, padding=0),
            nn.PixelShuffle(scale_factor)
        )
        self.head = nn.Sequential(
            Rearrange("b t c h w -> b c t h w"),
            ReflectionPad3d(1),
            nn.Conv3d(c, filters, kernel_size, stride=1, padding=kernel_size//2)
        )
        self.feature_attn = nn.Sequential(
            *[RFAB(filters, kernel_size, r) for _ in range(num_feature_attn_blocks)],
            nn.Conv3d(filters, filters, kernel_size, stride=1, padding=kernel_size//2)
        )
        self.temporal_redn = nn.Sequential(
            *[TemporalReductionBlock(filters, kernel_size, r) for _ in range(int(num_temporal_redn_blocks))]
        )
        self.main_upsample = nn.Sequential(
            nn.Conv3d(filters, c * scale_factor ** 2, kernel_size, stride=1, padding=0),
            Rearrange("b c t h w -> b (c t) h w"),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main branch
        h = self.head(x)
        feature_attn = h + self.feature_attn(h)
        temporal_redn = self.temporal_redn(feature_attn)
        temporal_redn = self.main_upsample(temporal_redn)

        # Global residual branch
        temporal_attn = self.temporal_attn(x)
        temporal_attn = self.residual_upsample(temporal_attn)

        return temporal_attn + temporal_redn
