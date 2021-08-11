import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange


class ResidualBlock(nn.Module):

    def __init__(channels: int = 64, kernel_size: int = 3):
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size//2),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class Encoder(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, num_res_blocks: int = 2):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2),
            *[ResidualBlock() for _ in range(num_res_blocks)],
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2),
        )


class CLSToken(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class Fusion(nn.Sequential):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 128,
        dropout: float = 0.1
    ):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        super().__init__(
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        )


class Pooling(nn.Module):

    def __init__(self, pool: str = "mean"):
        super().__init__()
        assert pool in ["mean", "cls"]
        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Decoder(nn.Sequential):

    def __init__(self, scale_factor: int = 3, in_channels: int = 64, kernel_size: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, channels * scale_factor ** 2, kernel_size, stride=1, padding=kernel_size//2),
            nn.PixelShuffle(scale_factor)
        )


class TRMISR(nn.Module):

    def __init__(
        self,
        scale_factor: int = 3,
        input_dim: int = 32,
        channels: int = 1,
        t: int = 9,
        emb_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        pool: str = "cls"
    ):
        super().__init__()
        output_dim = input_dim * scale_factor
        self.encoder = nn.Sequential(
            Rearrange("b t c h w -> (b t) c h w"),
            Encoder(channels, emb_dim),
            Rearrange("(b t) c () () -> b t c", t=t)
        )
        self.cls_token = CLSToken(emb_dim)
        self.fusion = Fusion(emb_dim, num_heads, num_layers)
        self.pool = Pooling(pool)
        self.decoder = nn.Sequential(
            Rearrange("b (h w) -> b () h w", h=output_dim, w=output_dim)
            Decoder(scale_factor, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = self.encoder(x)
        x = self.cls_token(x)
        x = self.fusion(x)
        x = self.pool(x)
        x = self.decoder(x)
        return x
