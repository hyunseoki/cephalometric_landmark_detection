import torch
import torch.nn as nn
from torch.nn import functional as F


### Convetional UNet ###
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False,
            **kwargs,
            )
        self.norm1 = nn.BatchNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False,
            **kwargs,
            )
        self.norm2 = nn.BatchNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            **kwargs,
        )
        self.norm = nn.BatchNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def max_pool():
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, n_filters=64):
        super().__init__()
        self.out_channels = out_channels

        ## encoder
        self.left1 = ConvBlock(in_channels=in_channels, out_channels=n_filters)
        self.left2 = nn.Sequential(
            max_pool(),
            ConvBlock(in_channels=n_filters, out_channels=n_filters * 2)
        )        
        self.left3 = nn.Sequential(
            max_pool(),
            ConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 4),
        )

        ## center
        self.center = nn.Sequential(
            max_pool(),
            ConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 8),
        )
        
        ## decoder
        self.up3 = UpsampleBlock(in_channels=n_filters * 8, out_channels=n_filters * 4)
        self.right3 = ConvBlock(in_channels=n_filters * 8, out_channels=n_filters * 4)
        self.up2 = UpsampleBlock(in_channels=n_filters * 4, out_channels=n_filters * 2)
        self.right2 = ConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 2)
        self.up1 = UpsampleBlock(in_channels=n_filters * 2, out_channels=n_filters * 1)
        self.right1 = ConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 1)

        ## score
        self.score = nn.Conv2d(
            in_channels=n_filters * 1, 
            out_channels=self.out_channels,
            kernel_size=1,
        )


    def forward(self, x):
        left1 = self.left1(x)
        left2 = self.left2(left1)
        left3 = self.left3(left2)

        center = self.center(left3)

        x = self.right3(torch.cat([self.up3(center), left3], 1))
        x = self.right2(torch.cat([self.up2(x), left2], 1))
        x = self.right1(torch.cat([self.up1(x), left1], 1))
        x = self.score(x)      

        if self.out_channels == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


### Squeeze-and-Excitation UNet ###
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//reduction_ratio)
        self.fc2 = nn.Linear(in_channels//reduction_ratio, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avgpool(x).view(b, c)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)

        return x * w.expand_as(x)


class SEUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, n_filters=64, reduction_ratio=16):
        super().__init__()
        assert n_filters > reduction_ratio        
        self.out_channels = out_channels

        ## encoder
        self.left1 = ConvBlock(in_channels=in_channels, out_channels=n_filters)
        self.left2 = nn.Sequential(
            max_pool(),
            ConvBlock(in_channels=n_filters, out_channels=n_filters * 2),
            SEBlock(in_channels=n_filters * 2, reduction_ratio=reduction_ratio),
        )        
        self.left3 = nn.Sequential(
            max_pool(),
            ConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 4),
            SEBlock(in_channels=n_filters * 4, reduction_ratio=reduction_ratio),
        )

        ## center
        self.center = nn.Sequential(
            max_pool(),
            ConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 8),
            SEBlock(in_channels=n_filters * 8, reduction_ratio=reduction_ratio),
        )
        
        ## decoder
        self.up3 = UpsampleBlock(in_channels=n_filters * 8, out_channels=n_filters * 4)
        self.right3 = nn.Sequential(
            ConvBlock(in_channels=n_filters * 8, out_channels=n_filters * 4),
            SEBlock(in_channels=n_filters * 4, reduction_ratio=reduction_ratio),
        )
        self.up2 = UpsampleBlock(in_channels=n_filters * 4, out_channels=n_filters * 2)
        self.right2 = nn.Sequential(
            ConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 2),
            SEBlock(in_channels=n_filters * 2, reduction_ratio=reduction_ratio),
        )
        self.up1 = UpsampleBlock(in_channels=n_filters * 2, out_channels=n_filters * 1)
        self.right1 = nn.Sequential(
            ConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 1),
            SEBlock(in_channels=n_filters * 1, reduction_ratio=reduction_ratio),
        )
        
        ## score
        self.score = nn.Conv2d(
            in_channels=n_filters * 1, 
            out_channels=self.out_channels,
            kernel_size=1,
        )


    def forward(self, x):
        left1 = self.left1(x)
        left2 = self.left2(left1)
        left3 = self.left3(left2)

        center = self.center(left3)

        x = self.right3(torch.cat([self.up3(center), left3], 1))
        x = self.right2(torch.cat([self.up2(x), left2], 1))
        x = self.right1(torch.cat([self.up1(x), left1], 1))
        x = self.score(x)      

        if self.out_channels == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


if __name__ == '__main__':
    # model = UNet(n_filters=1)
    # x = torch.randn(size=(3, 1, 16, 16), device='cpu')
    # print(model(x).shape)

    model = SEUNet()
    x = torch.randn(size=(3, 1, 16, 16), device='cpu')
    print(model(x).shape)