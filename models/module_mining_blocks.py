"""Retained SAD-Stem implementation extracted from the frozen candidate library."""
import torch
from torch import nn
import torch.nn.functional as F


def _gn(channels):
    groups = 4 if channels % 4 == 0 else (2 if channels % 2 == 0 else 1)
    return nn.GroupNorm(groups, channels)


class DynamicDepthwiseStem(nn.Module):
    """ODConv-inspired sample-wise 3x3/5x5 lightweight kernel routing."""
    def __init__(self, channels):
        super().__init__()
        self.k3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.k5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.router = nn.Conv2d(channels, 2, 1, bias=True)
        self.project = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), _gn(channels))

    def forward(self, x):
        weight = torch.softmax(self.router(F.adaptive_avg_pool2d(x, 1)), dim=1)
        z = weight[:, 0:1] * self.k3(x) + weight[:, 1:2] * self.k5(x)
        return x + self.project(z)
