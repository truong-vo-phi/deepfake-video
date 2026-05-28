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
        self.residual = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x, a_norm):
        res = self.residual(x)
        x = self.spatial_proj(x)
        x = torch.einsum("bctv,vw->bctw", x, a_norm)
        x = self.temporal_conv(x)
        x = self.bn(x)
        x = self.drop(x)
        return self.act(x + res)


class STGCNStream(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.block1 = STGCNBlock(in_channels, 64, dropout=dropout)
        self.block2 = STGCNBlock(64, hidden, dropout=dropout)

    def forward(self, x, a_norm):
        x = self.block1(x, a_norm)
        x = self.block2(x, a_norm)
        return x.mean(dim=(2, 3))  # [B, hidden]


class MultiStreamSTGCNClassifier(nn.Module):
    """
    Input channels expected from your current pipeline:
    [x, y, z, score, dx, dy, dz, speed] => C = 8
    """

    def __init__(self, in_channels: int = 8, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        if in_channels != 8:
            raise ValueError("Multi-stream model expects 8 input channels from stage03 features.")

        self.coord_stream = STGCNStream(in_channels=4, hidden=128, dropout=dropout)
        self.motion_stream = STGCNStream(in_channels=4, hidden=128, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, a_norm):
        # x: [B, 8, T, V]
        x_coord = x[:, :4, :, :]
        x_motion = x[:, 4:, :, :]
        f_coord = self.coord_stream(x_coord, a_norm)
        f_motion = self.motion_stream(x_motion, a_norm)
        f = torch.cat([f_coord, f_motion], dim=1)
        return self.head(f)
