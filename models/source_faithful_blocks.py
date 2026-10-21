"""Retained 2-D PSMamba transition (GFC-Mamba/SMB in the manuscript/code)."""
import torch
from torch import nn
from mamba_ssm import Mamba


class PSMambaLayer2D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        batch, channels = x.shape[:2]
        skip = x
        assert channels == self.dim
        image_dims = x.shape[2:]
        tokens = x.shape[2:].numel()
        flat = self.norm(x.reshape(batch, channels, tokens).transpose(-1, -2))
        out = self.mamba(flat).transpose(-1, -2).reshape(batch, channels, *image_dims)
        return out + skip


class PSSFE2D(nn.Module):
    """PSMamba SFE with only Conv3d/IN3d converted to Conv2d/IN2d."""
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(channels)
        self.nonliner = nn.ReLU()
        self.proj2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.nonliner2 = nn.ReLU()
        self.proj3 = nn.Conv2d(channels // 2, channels // 2, 3, 1, 1)
        self.norm3 = nn.InstanceNorm2d(channels)
        self.nonliner3 = nn.ReLU()
        self.proj4 = nn.Conv2d(channels // 2, channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm2d(channels)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        residual = x
        x1 = self.nonliner(self.norm(self.proj(x)))
        x1 = self.nonliner2(self.norm2(self.proj2(x1)))
        u, v = x1.chunk(2, dim=1)
        u = self.nonliner3(self.norm3(self.proj3(u)))
        x = u * v
        x = self.nonliner4(self.norm4(self.proj4(x)))
        return x + residual


class PSMlpChannel2D(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PSMambaBlock2D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = PSMambaLayer2D(dim, d_state, d_conv, expand)
        self.sfe = PSSFE2D(dim)
        self.norm = nn.InstanceNorm2d(dim)
        self.mlp = PSMlpChannel2D(dim, 2 * dim)

    def forward(self, x):
        x = self.sfe(x)
        x = self.mamba(x)
        x = self.norm(x)
        return self.mlp(x)


class PSConvInsBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()
        self.main = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        return self.activation(self.norm(self.main(x)))


class PSMambaTransition2D(nn.Module):
    """Source decoder SMB mapped to the Official one-input channel-transition API."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.smb = PSMambaBlock2D(input_dim)
        self.channel_transition = PSConvInsBlock2D(input_dim, output_dim)

    def forward(self, x):
        return self.channel_transition(self.smb(x))
