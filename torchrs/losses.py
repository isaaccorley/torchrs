import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    """ 
    Elementwise multiply pixelwise MSE loss with a binary mask
    Useful for ignoring some pixels given a quality mask
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='None')

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.mean(x * mask)
