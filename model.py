"""
MobileNetV3-UNet Semantic Segmentation Architecture.
Utilizes pretrained ImageNet weights for features and applies a lightweight UNet decoder.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights

class ConvBlock(nn.Module):
    """Dual Conv + BatchNorm + ReLU Block used in UNet Decoder steps."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class MobileNetV3UNet(nn.Module):
    """
    MobileNetV3-Large Encoder mapped to a custom lightweight decoder with skip connections.
    """
    def __init__(self, num_classes: int = 10, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Load Backbone
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        features = backbone.features
        
        # Optionally Freeze Backbone
        if freeze_backbone:
            for param in features.parameters():
                param.requires_grad = False
                
        # Split features for intermediate multi-scale Skip Connections
        self.enc0 = features[0:2]    # Output shape: [B, 16, H/2, W/2]
        self.enc1 = features[2:4]    # Output shape: [B, 24, H/4, W/4]
        self.enc2 = features[4:7]    # Output shape: [B, 40, H/8, W/8]
        self.enc3 = features[7:13]   # Output shape: [B, 112, H/16, W/16]
        self.enc4 = features[13:17]  # Output shape: [B, 960, H/32, W/32]
        
        # UNet Decoder Blocks
        self.up4 = nn.ConvTranspose2d(960, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256 + 112, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 40, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32 + 24, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16 + 16, 16)
        
        self.up0 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder Feed Forward & Extraction
        s0 = self.enc0(x)     # H/2 (16 ch)
        s1 = self.enc1(s0)    # H/4 (24 ch)
        s2 = self.enc2(s1)    # H/8 (40 ch)
        s3 = self.enc3(s2)    # H/16 (112 ch)
        s4 = self.enc4(s3)    # H/32 (960 ch)
        
        # Decoder Feed Forward with Skip Connection Concatenation
        x = self.up4(s4)
        x = torch.cat([x, s3], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, s0], dim=1)
        x = self.dec1(x)
        
        x = self.up0(x)
        x = self.final_conv(x)
        
        return x
