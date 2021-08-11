import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class Encoder(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        )


class CLSToken(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class Fusion(nn.Sequential):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
    ):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            activation="gelu"
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

    def __init__(self, patch_size: int):
        super().__init__(
            Rearrange("b (h w) -> b () h w", h=patch_size, w=patch_size)
        )


class TRMISR(nn.Module):

    def __init__(
        self,
        scale_factor: int = 3,
        input_dim: int = 32,
        channels: int = 1,
        t: int = 9,
        num_layers: int = 8,
        num_heads: int = 16,
        pool: str = "cls"
    ):
        super().__init__()
        emb_dim = (input_dim ** 2) * scale_factor
        self.encoder = nn.Sequential(
            Rearrange("b t c h w -> (b t) c h w"),
            Encoder(in_channels=channels, out_channels=emb_dim, kernel_size=input_dim),
            Rearrange("(b t) c -> b t c", t=t)
        )
        self.cls_token = CLSToken(emb_dim)
        self.fusion = Fusion(emb_dim, num_heads, num_layers)
        self.pool = Pooling(pool)
        self.decoder = Decoder(patch_size=input_dim * scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = self.encoder(x)
        x = self.cls_token(x)
        x = self.fusion(x)
        x = self.pool(x)
        x = self.decoder(x)
        return x
