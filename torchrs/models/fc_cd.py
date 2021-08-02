from typing import List, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class ConvBlock(nn.Module):

    def __init__(self, filters: List[int], kernel_size: int = 3, dropout: float = 0.2, pool: bool = True):
        super().__init__()
        layers = []
        for i in range(1, len(filters)):
            layers.extend([
                nn.Conv2d(filters[i - 1], filters[i], kernel_size, stride=1, padding=kernel_size//2),
                nn.BatchNorm2d(filters[i]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.model = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.model(x)
        return self.pool(x), x


class DeConvBlock(nn.Sequential):

    def __init__(self, filters: List[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__(
            *[nn.Sequential(
                nn.ConvTranspose2d(filters[i - 1], filters[i], kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(filters[i]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(1, len(filters))]
        )


class UpsampleBlock(nn.Sequential):

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__(
            nn.ConvTranspose2d(channels, channels, kernel_size, padding=kernel_size//2, stride=2, output_padding=1)
        )


class Encoder(nn.ModuleList):

    def __init__(self, in_channels: int = 3):
        super().__init__([
            ConvBlock([in_channels, 16, 16]),
            ConvBlock([16, 32, 32]),
            ConvBlock([32, 64, 64, 64]),
            ConvBlock([64, 128, 128, 128])
        ])


class Decoder(nn.ModuleList):

    def __init__(self, num_classes: int = 2):
        super().__init__([
            DeConvBlock([256, 128, 128, 64]),
            DeConvBlock([128, 64, 64, 32]),
            DeConvBlock([64, 32, 16]),
            DeConvBlock([32, 16, num_classes])
        ])


class SiamEncoder(nn.ModuleList):

    def __init__(self, in_channels: int = 3):
        super().__init__([
            ConvBlock([in_channels, 16, 16]),
            ConvBlock([16, 32, 32]),
            ConvBlock([32, 64, 64, 64]),
            ConvBlock([64, 128, 128, 128], pool=False)
        ])


class ConcatDecoder(nn.ModuleList):

    def __init__(self, t: int = 2, num_classes: int = 2):
        scale = 0.5 * (t + 1)
        super().__init__([
            DeConvBlock([int(256 * scale), 128, 128, 64]),
            DeConvBlock([int(128 * scale), 64, 64, 32]),
            DeConvBlock([int(64 * scale), 32, 16]),
            DeConvBlock([int(32 * scale), 16, num_classes])
        ])


class Upsample(nn.ModuleList):

    def __init__(self):
        super().__init__([
            UpsampleBlock(128),
            UpsampleBlock(64),
            UpsampleBlock(32),
            UpsampleBlock(16)
        ])


class FCEF(nn.Module):
    """ Fully-convolutional Early Fusion (FC-EF) from
    'Fully Convolutional Siamese Networks for Change Detection', Daudt et al. (2018)
    https://arxiv.org/abs/1810.08462
    """
    def __init__(self, channels: int = 3, t: int = 2, num_classes: int = 2):
        super().__init__()
        self.encoder = Encoder(channels * t)
        self.decoder = Decoder(num_classes)
        self.upsample = Upsample()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> b (t c) h w")

        skips = []
        for block in self.encoder:
            x, skip = block(x)
            skips.append(skip)

        for block, upsample, skip in zip(self.decoder, self.upsample, reversed(skips)):
            x = upsample(x)
            x = rearrange([x, skip], "t b c h w -> b (t c) h w")
            x = block(x)

        return x


class FCSiamConc(nn.Module):
    """ Fully-convolutional Siamese Concatenation (FC-Siam-conc) from
    'Fully Convolutional Siamese Networks for Change Detection', Daudt et al. (2018)
    https://arxiv.org/abs/1810.08462
    """
    def __init__(self, channels: int = 3, t: int = 2, num_classes: int = 2):
        super().__init__()
        self.encoder = SiamEncoder(channels)
        self.decoder = ConcatDecoder(t, num_classes)
        self.upsample = Upsample()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")

        skips = []
        for block in self.encoder:
            x, skip = block(x)
            skips.append(skip)

        # Concat skips
        skips = [rearrange(skip, "(b t) c h w -> b (t c) h w", t=t) for skip in skips]

        # Only first input encoding is passed directly to decoder
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)
        x = x[:, 0, ...]
        x = self.pool(x)

        for block, upsample, skip in zip(self.decoder, self.upsample, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x


class FCSiamDiff(nn.Module):
    """ Fully-convolutional Siamese Difference (FC-Siam-diff) from
    'Fully Convolutional Siamese Networks for Change Detection', Daudt et al. (2018)
    https://arxiv.org/abs/1810.08462
    """
    def __init__(self, channels: int = 3, t: int = 2, num_classes: int = 2):
        super().__init__()
        self.encoder = SiamEncoder(channels)
        self.decoder = Decoder(num_classes)
        self.upsample = Upsample()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")

        skips = []
        for block in self.encoder:
            x, skip = block(x)
            skips.append(skip)

        # Diff skips
        skips = [rearrange(skip, "(b t) c h w -> b t c h w", t=t) for skip in skips]
        diffs = []
        for skip in skips:
            diff, xt = skip[:, 0, ...], skip[:, 1:, ...]
            for i in range(t - 1):
                diff = diff - xt[:, i, ...]
                diffs.append(diff)

        # Only first input encoding is passed directly to decoder
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)
        x = x[:, 0, ...]
        x = self.pool(x)

        for block, upsample, skip in zip(self.decoder, self.upsample, reversed(diffs)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x
