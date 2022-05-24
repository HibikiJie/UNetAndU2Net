import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ConvolutionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1 * dilation,
                      dilation=(1 * dilation, 1 * dilation)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_s1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1 * dilation,
                                 dilation=(1 * dilation, 1 * dilation))
        self.bn_s1 = nn.BatchNorm2d(out_channels)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.layer(x)


def upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
    return src


class DownSample(nn.Module):

    def __init__(self, ):
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.layer(x)


class UNet1(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNet1, self).__init__()

        self.conv0 = ConvolutionLayer(in_channels, out_channels, dilation=1)

        self.conv1 = ConvolutionLayer(out_channels, mid_channels, dilation=1)
        self.down1 = DownSample()

        self.conv2 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down2 = DownSample()

        self.conv3 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down3 = DownSample()

        self.conv4 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down4 = DownSample()

        self.conv5 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down5 = DownSample()

        self.conv6 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)

        self.conv7 = ConvolutionLayer(mid_channels, mid_channels, dilation=2)

        self.conv8 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv9 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv10 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv11 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv12 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv13 = ConvolutionLayer(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(x0)
        d1 = self.down1(x1)

        x2 = self.conv2(d1)
        d2 = self.down2(x2)

        x3 = self.conv3(d2)
        d3 = self.down3(x3)

        x4 = self.conv4(d3)
        d4 = self.down4(x4)

        x5 = self.conv5(d4)
        d5 = self.down5(x5)

        x6 = self.conv6(d5)

        x7 = self.conv7(x6)

        x8 = self.conv8(torch.cat((x7, x6), 1))
        up1 = upsample_like(x8, x5)

        x9 = self.conv9(torch.cat((up1, x5), 1))
        up2 = upsample_like(x9, x4)

        x10 = self.conv10(torch.cat((up2, x4), 1))
        up3 = upsample_like(x10, x3)

        x11 = self.conv11(torch.cat((up3, x3), 1))
        up4 = upsample_like(x11, x2)

        x12 = self.conv12(torch.cat((up4, x2), 1))
        up5 = upsample_like(x12, x1)

        x13 = self.conv13(torch.cat((up5, x1), 1))

        return x13 + x0


class UNet2(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNet2, self).__init__()

        self.conv0 = ConvolutionLayer(in_channels, out_channels, dilation=1)

        self.conv1 = ConvolutionLayer(out_channels, mid_channels, dilation=1)
        self.down1 = DownSample()

        self.conv2 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down2 = DownSample()

        self.conv3 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down3 = DownSample()

        self.conv4 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down4 = DownSample()

        self.conv5 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)

        self.conv6 = ConvolutionLayer(mid_channels, mid_channels, dilation=2)

        self.conv7 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv8 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv9 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv10 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv11 = ConvolutionLayer(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(x0)
        d1 = self.down1(x1)

        x2 = self.conv2(d1)
        d2 = self.down2(x2)

        x3 = self.conv3(d2)
        d3 = self.down3(x3)

        x4 = self.conv4(d3)
        d4 = self.down4(x4)

        x5 = self.conv5(d4)

        x6 = self.conv6(x5)

        x7 = self.conv7(torch.cat((x6, x5), dim=1))
        up1 = upsample_like(x7, x4)

        x8 = self.conv8(torch.cat((up1, x4), dim=1))
        up2 = upsample_like(x8, x3)

        x9 = self.conv9(torch.cat((up2, x3), dim=1))
        up3 = upsample_like(x9, x2)

        x10 = self.conv10(torch.cat((up3, x2), dim=1))
        up4 = upsample_like(x10, x1)

        x11 = self.conv11(torch.cat((up4, x1), dim=1))

        return x11 + x0


class UNet3(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNet3, self).__init__()

        self.conv0 = ConvolutionLayer(in_channels, out_channels, dilation=1)

        self.conv1 = ConvolutionLayer(out_channels, mid_channels, dilation=1)
        self.down1 = DownSample()

        self.conv2 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down2 = DownSample()

        self.conv3 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down3 = DownSample()

        self.conv4 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)

        self.conv5 = ConvolutionLayer(mid_channels, mid_channels, dilation=2)

        self.conv6 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv7 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv8 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv9 = ConvolutionLayer(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(x0)
        d1 = self.down1(x1)

        x2 = self.conv2(d1)
        d2 = self.down2(x2)

        x3 = self.conv3(d2)
        d3 = self.down3(x3)

        x4 = self.conv4(d3)

        x5 = self.conv5(x4)

        x6 = self.conv6(torch.cat((x5, x4), 1))
        up1 = upsample_like(x6, x3)

        x7 = self.conv7(torch.cat((up1, x3), 1))
        up2 = upsample_like(x7, x2)

        x8 = self.conv8(torch.cat((up2, x2), 1))
        up3 = upsample_like(x8, x1)

        x9 = self.conv9(torch.cat((up3, x1), 1))

        return x9 + x0


class UNet4(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNet4, self).__init__()

        self.conv0 = ConvolutionLayer(in_channels, out_channels, dilation=1)

        self.conv1 = ConvolutionLayer(out_channels, mid_channels, dilation=1)
        self.down1 = DownSample()

        self.conv2 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)
        self.down2 = DownSample()

        self.conv3 = ConvolutionLayer(mid_channels, mid_channels, dilation=1)

        self.conv4 = ConvolutionLayer(mid_channels, mid_channels, dilation=2)

        self.conv5 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv6 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=1)
        self.conv7 = ConvolutionLayer(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        """encode"""
        x0 = self.conv0(x)

        x1 = self.conv1(x0)
        d1 = self.down1(x1)

        x2 = self.conv2(d1)
        d2 = self.down2(x2)

        x3 = self.conv3(d2)

        x4 = self.conv4(x3)
        """decode"""
        x5 = self.conv5(torch.cat((x4, x3), 1))
        up1 = upsample_like(x5, x2)

        x6 = self.conv6(torch.cat((up1, x2), 1))
        up2 = upsample_like(x6, x1)

        x7 = self.conv7(torch.cat((up2, x1), 1))

        return x7 + x0


class UNet5(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNet5, self).__init__()

        self.conv0 = ConvolutionLayer(in_channels, out_channels, dilation=1)

        self.conv1 = ConvolutionLayer(out_channels, mid_channels, dilation=1)
        self.conv2 = ConvolutionLayer(mid_channels, mid_channels, dilation=2)
        self.conv3 = ConvolutionLayer(mid_channels, mid_channels, dilation=4)

        self.conv4 = ConvolutionLayer(mid_channels, mid_channels, dilation=8)

        self.conv5 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=4)
        self.conv6 = ConvolutionLayer(mid_channels * 2, mid_channels, dilation=2)
        self.conv7 = ConvolutionLayer(mid_channels * 2, out_channels, dilation=1)

    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x4 = self.conv4(x3)

        x5 = self.conv5(torch.cat((x4, x3), 1))
        x6 = self.conv6(torch.cat((x5, x2), 1))
        x7 = self.conv7(torch.cat((x6, x1), 1))

        return x7 + x0


class U2Net(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(U2Net, self).__init__()

        self.en_1 = UNet1(in_channels, 32, 64)
        self.down1 = DownSample()

        self.en_2 = UNet2(64, 32, 128)
        self.down2 = DownSample()

        self.en_3 = UNet3(128, 64, 256)
        self.down3 = DownSample()

        self.en_4 = UNet4(256, 128, 512)
        self.down4 = DownSample()

        self.en_5 = UNet5(512, 256, 512)
        self.down5 = DownSample()

        self.en_6 = UNet5(512, 256, 512)

        # decoder
        self.de_5 = UNet5(1024, 256, 512)
        self.de_4 = UNet4(1024, 128, 256)
        self.de_3 = UNet3(512, 64, 128)
        self.de_2 = UNet2(256, 32, 64)
        self.de_1 = UNet1(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1)
        self.side2 = nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1)
        self.side3 = nn.Conv2d(128, out_channels, kernel_size=(3, 3), padding=1)
        self.side4 = nn.Conv2d(256, out_channels, kernel_size=(3, 3), padding=1)
        self.side5 = nn.Conv2d(512, out_channels, kernel_size=(3, 3), padding=1)
        self.side6 = nn.Conv2d(512, out_channels, kernel_size=(3, 3), padding=1)

        self.out_conv = nn.Conv2d(6, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        # ------encode ------
        x1 = self.en_1(x)
        d1 = self.down1(x1)

        x2 = self.en_2(d1)
        d2 = self.down2(x2)

        x3 = self.en_3(d2)
        d3 = self.down3(x3)

        x4 = self.en_4(d3)
        d4 = self.down4(x4)

        x5 = self.en_5(d4)
        d5 = self.down5(x5)

        x6 = self.en_6(d5)
        up1 = upsample_like(x6, x5)

        # ------decode ------
        x7 = self.de_5(torch.cat((up1, x5), dim=1))
        up2 = upsample_like(x7, x4)

        x8 = self.de_4(torch.cat((up2, x4), dim=1))
        up3 = upsample_like(x8, x3)

        x9 = self.de_3(torch.cat((up3, x3), dim=1))
        up4 = upsample_like(x9, x2)

        x10 = self.de_2(torch.cat((up4, x2), dim=1))
        up5 = upsample_like(x10, x1)

        x11 = self.de_1(torch.cat((up5, x1), dim=1))

        # side output
        sup1 = self.side1(x11)

        sup2 = self.side2(x10)
        sup2 = upsample_like(sup2, sup1)

        sup3 = self.side3(x9)
        sup3 = upsample_like(sup3, sup1)

        sup4 = self.side4(x8)
        sup4 = upsample_like(sup4, sup1)

        sup5 = self.side5(x7)
        sup5 = upsample_like(sup5, sup1)

        sup6 = self.side6(x6)
        sup6 = upsample_like(sup6, sup1)

        sup0 = self.out_conv(torch.cat((sup1, sup2, sup3, sup4, sup5, sup6), 1))

        return torch.sigmoid(sup0)


if __name__ == '__main__':
    u2net = U2Net(3, 1)
    x = torch.randn(1,3, 512, 512)
    print(u2net(x).shape)
