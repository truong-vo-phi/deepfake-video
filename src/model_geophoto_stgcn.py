import torch
import torch.nn as nn


class STGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_kernel: int = 9, dropout: float = 0.2):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.spatial_proj = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.temporal_conv = nn.Conv2d(c_out, c_out, kernel_size=(t_kernel, 1), padding=(pad, 0))
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


class GeoEncoder(nn.Module):
    def __init__(self, in_channels=8, dropout=0.2):
        super().__init__()
        self.block1 = STGCNBlock(in_channels, 64, dropout=dropout)
        self.block2 = STGCNBlock(64, 128, dropout=dropout)
        self.block3 = STGCNBlock(128, 256, dropout=dropout)

    def forward(self, x, a_norm):
        x = self.block1(x, a_norm)
        x = self.block2(x, a_norm)
        x = self.block3(x, a_norm)
        return x.mean(dim=(2, 3))  # [B,256]


class PhotoEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x_photo):
        # x_photo: [B, Tp, 3, H, W]
        b, t, c, h, w = x_photo.shape
        x = x_photo.reshape(b * t, c, h, w)
        x = self.cnn(x).reshape(b * t, -1)
        x = self.proj(x).reshape(b, t, -1)  # [B,Tp,D]
        return x.mean(dim=1)  # [B,D]


class GeoPhotoSTGCN(nn.Module):
    def __init__(self, in_channels=8, num_classes=2, dim=256, heads=4, dropout=0.2):
        super().__init__()
        self.geo = GeoEncoder(in_channels=in_channels, dropout=dropout)
        self.photo = PhotoEncoder(out_dim=dim)
        self.geo_to_photo = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.photo_to_geo = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(dim * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_geo, x_photo, a_norm):
        fg = self.geo(x_geo, a_norm)      # [B,D]
        fp = self.photo(x_photo)          # [B,D]
        tg = fg.unsqueeze(1)              # [B,1,D]
        tp = fp.unsqueeze(1)              # [B,1,D]
        g_from_p, _ = self.geo_to_photo(query=tg, key=tp, value=tp)
        p_from_g, _ = self.photo_to_geo(query=tp, key=tg, value=tg)
        fused = torch.cat([fg, fp, g_from_p.squeeze(1), p_from_g.squeeze(1)], dim=1)
        return self.head(fused)
