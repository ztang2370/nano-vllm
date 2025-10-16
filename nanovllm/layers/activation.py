import torch
from torch import nn
import torch.nn.functional as F


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


class ReplicaSiluAndMul(nn.Module):
    """SiLU activation without torch.compile for replication testing."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
