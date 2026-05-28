import torch
import torch.nn as nn


class STGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_kernel: int = 9, dropout: float = 0.2):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.spatial_proj = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.temporal_conv = nn.Conv2d(
            c_out,
            c_out,
            kernel_size=(t_kernel, 1),
            padding=(pad, 0),
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        if c_in == c_out:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x, a_norm):
        # x: [B, C, T, V], a_norm: [V, V]
        res = self.residual(x)
        x = self.spatial_proj(x)
        x = torch.einsum("bctv,vw->bctw", x, a_norm)
        x = self.temporal_conv(x)
        x = self.bn(x)
        x = self.drop(x)
        return self.act(x + res)


class STGCNClassifier(nn.Module):
    def __init__(self, in_channels: int = 8, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.block1 = STGCNBlock(in_channels, 64, dropout=dropout)
        self.block2 = STGCNBlock(64, 128, dropout=dropout)
        self.block3 = STGCNBlock(128, 256, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, a_norm):
        x = self.block1(x, a_norm)
        x = self.block2(x, a_norm)
        x = self.block3(x, a_norm)
        x = x.mean(dim=(2, 3))  # global average pooling over T,V
        return self.head(x)
