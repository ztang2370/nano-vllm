import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class SimpleSilu(nn.Module):
    """Simple SiLU activation for replication testing (no torch.compile)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)
