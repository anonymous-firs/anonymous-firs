import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SoftHistogram(nn.Module):
    def __init__(self, bins: int = 32, min: float = 0.0, max: float = 1.0, sigma: float = 0.01):
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = (max - min) / bins
        centers = min + self.delta * (0.5 + torch.arange(bins))
        self.register_buffer("centers", centers)  # [bins]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, NP, P], where NP is the number of patches and P is patch area.
        return: [B, C, NP, bins]
        """
        # -> [B, C, NP, P, 1]
        x = x.unsqueeze(-1)
        centers = self.centers.to(x.device)  # [bins]
        # -> [B, C, NP, P, bins]
        w = torch.exp(-0.5 * ((x - centers) / (self.sigma + 1e-12)) ** 2)

        hist = w.mean(dim=-2)  # [B, C, NP, bins]
        return hist


# -----------------------------
# Grid-aware distribution stem
# -----------------------------
class GridDistributionStem(nn.Module):
    """Tile-wise Statistical Encoder (TSE).

    The encoder partitions each image into a regular grid, computes soft
    intensity histograms and low-order moments per tile, and projects the
    resulting statistics into a compact embedding.
    """

    def __init__(self, out_dim: int = 128, bins: int = 32, sigma: float = 0.05, grid_size: int = -1):
        """
        grid_size: split HxW into a grid_size x grid_size grid of non-overlapping patches.
                   If -1, choose grid size automatically based on input size:
                       min(H,W) >= 64 -> 8
                       min(H,W) >= 32 -> 4
                       else           -> 2
        """
        super().__init__()
        self.grid_size = grid_size
        self.hist_layer = SoftHistogram(bins=bins, sigma=sigma)
        self.bins = bins
        self._proj = None
        self.out_dim = out_dim

    def _auto_grid(self, H: int, W: int) -> int:
        m = min(H, W)
        if m >= 64:
            return 8
        if m >= 32:
            return 4
        return 2

    def _ensure_proj(self, in_dim: int, device: torch.device):
        """Create the projection layer once and place it on the correct device."""
        if self._proj is None:
            self._proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 256),
                nn.GELU(),
                nn.Linear(256, self.out_dim)
            ).to(device)

    def _extract_grid(self, x: torch.Tensor, g: int) -> torch.Tensor:
        """
        x: [B, 3, H, W], with pixel values in [0, 1].
        return: x_grid [B,3,NP,P]  (NP=g*g, P=patch_area)
        """
        B, C, H, W = x.shape

        if (H % g != 0) or (W % g != 0):
            new_h = (H // g + (1 if H % g != 0 else 0)) * g
            new_w = (W // g + (1 if W % g != 0 else 0)) * g
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            B, C, H, W = x.shape

        ph, pw = H // g, W // g
        # unfold -> [B, C*ph*pw, NP]
        patches = F.unfold(x, kernel_size=(ph, pw), stride=(ph, pw))
        NP = patches.shape[-1]
        P = ph * pw
        # -> [B, C, NP, P]
        x_grid = patches.view(B, C, P, NP).permute(0, 1, 3, 2).contiguous()
        return x_grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] in [0,1]
        return: dist_feat [B, out_dim]
        """
        B, C, H, W = x.shape
        g = self.grid_size if self.grid_size != -1 else self._auto_grid(H, W)
        assert g >= 2, "grid_size must be >= 2"


        x_grid = self._extract_grid(x, g)        # [B,3,NP,P]
        B, C, NP, P = x_grid.shape

        hist = self.hist_layer(x_grid)           # [B,3,NP,bins]
        mean = x_grid.mean(-1)                   # [B,3,NP]
        variance = x_grid.var(-1, unbiased=False) + 1e-6
        std = torch.sqrt(variance)
        z = (x_grid - mean[..., None]) / std[..., None]
        skew = (z ** 3).mean(-1)                 # [B,3,NP]
        kurt = (z ** 4).mean(-1) - 3.0           # [B,3,NP]


        grid_feat = torch.cat([
            hist,                     # [B,3,NP,bins]
            mean[..., None],          # [B,3,NP,1]
            variance[..., None],      # [B,3,NP,1]
            skew[..., None],          # [B,3,NP,1]
            kurt[..., None],          # [B,3,NP,1]
        ], dim=3).contiguous()        # [B,3,NP,(bins+4)]


        grid_feat = grid_feat.view(B, -1)
        in_dim = NP * 3 * (self.bins + 4)


        self._ensure_proj(in_dim, x.device)

        return self._proj(grid_feat)             # [B, out_dim]


# -----------------------------
# ResNet18 + grid distribution stem
# -----------------------------
class ResNet18Dist(nn.Module):
    """FIRS detector with GSE, TSE, and Fusion Screening Head (FSH)."""

    def __init__(self, pretrained: bool = True, bins: int = 32, dist_dim: int = 128, grid_size: int = -1):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.backbone.fc = nn.Identity()
        self.feat_dim = 512


        self.dist_stem = GridDistributionStem(out_dim=dist_dim, bins=bins, sigma=0.05, grid_size=grid_size)
        self.dist_dim = dist_dim


        self.fusion_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim + self.dist_dim),
            nn.Linear(self.feat_dim + self.dist_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        self.head = self.fusion_head


        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Returns:
          logits: [B], compatible with BCEWithLogitsLoss targets shaped [B].
          dist_feat: [B, dist_dim], used by the supervised contrastive term.
        """

        dist_feat = self.dist_stem(x)  # [B, dist_dim]


        x_norm = (x - self.im_mean.to(x.device)) / self.im_std.to(x.device)
        cnn_feat = self.backbone(x_norm)  # [B, 512]


        fused_feat = torch.cat([cnn_feat, dist_feat], dim=1)      # [B, 512+dist_dim]
        logits = self.fusion_head(fused_feat).squeeze(-1)
        if return_features:
            return {
                "logits": logits,
                "semantic_features": cnn_feat,
                "statistical_features": dist_feat,
                "fused_features": fused_feat,
            }
        return logits, dist_feat
