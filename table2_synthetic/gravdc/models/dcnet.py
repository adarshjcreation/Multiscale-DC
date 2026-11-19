"""DC-Net baseline (Li et al., 2023): a 5-level U-Net with batch normalisation."""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DCNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.enc1 = ConvBlock(32, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = ConvBlock(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.postprocess = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        down1 = self.preprocess(x)
        down2 = self.enc1(self.pool(down1))
        down3 = self.enc2(self.pool(down2))
        down4 = self.enc3(self.pool(down3))
        down5 = self.enc4(self.pool(down4))
        enc5 = self.enc5(self.pool(down5))

        x = self.dec5(torch.cat([self.up5(enc5), down5], dim=1))
        x = self.dec4(torch.cat([self.up4(x), down4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), down3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), down2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), down1], dim=1))
        return self.postprocess(x)
