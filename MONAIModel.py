import torch
import monai
from monai.networks.blocks import Convolution

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = torch.nn.Sequential(
            Convolution(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                act='RELU'
            ),
            Convolution(
                spatial_dims=2,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                act='RELU'
            )
        )
        
    def forward(self, X):
        return self.step(X)
    
    
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = DoubleConv(in_channels=1, out_channels=64)
        self.layer2 = DoubleConv(in_channels=64, out_channels=128)
        self.layer3 = DoubleConv(in_channels=128, out_channels=256)
        self.layer4 = DoubleConv(in_channels=256, out_channels=512)

        self.layer5 = DoubleConv(in_channels=512+256, out_channels=256)
        self.layer6 = DoubleConv(in_channels=256+128, out_channels=128)
        self.layer7 = DoubleConv(in_channels=128+64, out_channels=64)
        self.layer8 = Convolution(spatial_dims=2, in_channels=64, out_channels=1, kernel_size=1)

        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, X):
        x1 = self.layer1(X)
        x1m = self.maxpool(x1)

        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)

        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)

        x4 = self.layer4(x3m)

        x5 = monai.networks.blocks.Upsample(spatial_dims=2, scale_factor=2, mode="nontrainable")(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)

        x6 = monai.networks.blocks.Upsample(spatial_dims=2, scale_factor=2, mode="nontrainable")(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)

        x7 = monai.networks.blocks.Upsample(spatial_dims=2, scale_factor=2, mode="nontrainable")(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)

        ret = self.layer8(x7)

        return ret
