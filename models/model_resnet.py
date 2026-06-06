
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models

class SoftHistogram(nn.Module):
    def __init__(self, bins=32, sigma=0.05):
        super().__init__()
        self.bins = bins
        centers = torch.linspace(0.0, 1.0, bins).view(1, 1, bins)
        self.register_buffer("centers", centers)
        self.sigma = sigma

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.view(B, C, -1).unsqueeze(-1)
        dist2 = (x_flat - self.centers)**2
        w = torch.exp(- dist2 / (2 * self.sigma**2))

        w = w / (w.sum(dim=2, keepdim=True) + 1e-9)
        hist = w.sum(dim=2) / (H*W)
        return hist


class DistributionStem(nn.Module):
    def __init__(self, out_dim=128, bins=32, sigma=0.05):
        super().__init__()
        self.hist = SoftHistogram(bins=bins, sigma=sigma)
        self.proj = nn.Sequential(
            nn.LayerNorm(3*(bins + 4)),
            nn.Linear(3*(bins + 4), 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        mean = x_flat.mean(-1)
        std  = x_flat.std(-1) + 1e-6
        z = (x_flat - mean[...,None]) / std[...,None]
        skew = (z**3).mean(-1)
        kurt = (z**4).mean(-1) - 3.0
        hist = self.hist(x)
        feat = torch.cat([hist.view(B, -1), mean, std, skew, kurt], dim=1)
        return self.proj(feat)

class ResNet18Dist(nn.Module):
    def __init__(self, pretrained=True, bins=16, dist_dim=128):
        super().__init__()


        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.feat_dim = 512

        self.dist_stem = DistributionStem(out_dim=dist_dim, bins=bins, sigma=0.05)
        self.dist_dim = dist_dim

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim + self.dist_dim),
            nn.Linear(self.feat_dim + self.dist_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x_img):

        mean = torch.tensor([0.485, 0.456, 0.406], device=x_img.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x_img.device).view(1,3,1,1)
        x_norm = (x_img - mean) / std

        cnn_feat = self.backbone(x_norm)
        dist_feat = self.dist_stem(x_img)

        fused = torch.cat([cnn_feat, dist_feat], dim=1)

        logit = self.head(fused).squeeze(-1)
        return logit, dist_feat

class ResNet18Only(nn.Module):
    """
    Binary classifier head built on ResNet18.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        if pretrained:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Linear(in_dim, 1)
        self.net = backbone

    def forward(self, x):
        logits = self.net(x).squeeze(1)                 # [B]
        return logits, None
