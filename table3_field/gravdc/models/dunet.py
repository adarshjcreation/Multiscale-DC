"""DU-Net baseline (Ye et al., 2022): a U-Net built from dense blocks."""
import torch
from torch import nn


def crop_to_match(enc, dec):
    """Centre-crop the encoder tensor to the decoder's spatial size."""
    _, _, h_enc, w_enc = enc.size()
    _, _, h_dec, w_dec = dec.size()
    crop_h = (h_enc - h_dec) // 2
    crop_w = (w_enc - w_dec) // 2
    return enc[:, :, crop_h:crop_h + h_dec, crop_w:crop_w + w_dec]


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
            ))
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        out_channels = int(in_channels * reduction)
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.transition(x)


class DUnet(nn.Module):
    def __init__(self, input_channels=1, growth_rate=16, num_layers=4):
        super().__init__()
        self.encoder1 = DenseBlock(input_channels, growth_rate, num_layers)
        self.trans1 = TransitionLayer(self.encoder1.out_channels, 0.5)
        self.encoder2 = DenseBlock(self.trans1.out_channels, growth_rate * 2, num_layers)
        self.trans2 = TransitionLayer(self.encoder2.out_channels, 0.5)
        self.encoder3 = DenseBlock(self.trans2.out_channels, growth_rate * 4, num_layers)

        self.bottleneck = DenseBlock(self.encoder3.out_channels, growth_rate * 8, num_layers)

        self.upconv1 = nn.ConvTranspose2d(self.bottleneck.out_channels, growth_rate * 4, kernel_size=3, stride=1,
                                          padding=1)
        self.decoder1 = DenseBlock(self.encoder3.out_channels + growth_rate * 4, growth_rate * 4, num_layers)
        self.upconv2 = nn.ConvTranspose2d(self.decoder1.out_channels, growth_rate * 2, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.decoder2 = DenseBlock(self.encoder2.out_channels + growth_rate * 2, growth_rate * 2, num_layers)
        self.upconv3 = nn.ConvTranspose2d(self.decoder2.out_channels, growth_rate, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.decoder3 = DenseBlock(self.encoder1.out_channels + growth_rate, growth_rate, num_layers)

        self.output_layer = nn.Conv2d(self.decoder3.out_channels, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.trans1(enc1))
        enc3 = self.encoder3(self.trans2(enc2))
        x = self.bottleneck(enc3)

        x = self.upconv1(x)
        x = self.decoder1(torch.cat([x, crop_to_match(enc3, x)], dim=1))
        x = self.upconv2(x)
        x = self.decoder2(torch.cat([x, crop_to_match(enc2, x)], dim=1))
        x = self.upconv3(x)
        x = self.decoder3(torch.cat([x, crop_to_match(enc1, x)], dim=1))
        return self.output_layer(x)
