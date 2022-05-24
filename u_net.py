from torch import nn
import torch


class ConvolutionLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        卷积层
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        """
        super(ConvolutionLayer, self).__init__()
        self.layer = nn.Sequential(
            # 卷积层
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # BN层
            nn.ReLU(),  # 激活
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):

    def __init__(self, ):
        """
        最大池化层构成的下采样，池化窗口为2×2
        """
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):

    def __init__(self, in_channels):
        """
        反卷积，上采样
        :param in_channels: 输入通道数
        """
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        """
        前向运算过程
        :param input_: 输入
        :param concat: 浅层特征
        :return: 完成上采样后和浅层特征concat的结果
        """
        return self.layer(x)


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        UNet网络，结构类似编解码，
        """
        super(UNet, self).__init__()
        self.conv1 = ConvolutionLayer(in_channels, 64)
        self.down1 = DownSample()
        self.conv2 = ConvolutionLayer(64, 128)
        self.down2 = DownSample()
        self.conv3 = ConvolutionLayer(128, 256)
        self.down3 = DownSample()
        self.conv4 = ConvolutionLayer(256, 512)
        self.down4 = DownSample()
        self.conv5 = ConvolutionLayer(512, 1024)
        self.up1 = UpSample(1024)
        self.conv6 = ConvolutionLayer(1024, 512)
        self.up2 = UpSample(512)
        self.conv7 = ConvolutionLayer(512, 256)
        self.up3 = UpSample(256)
        self.conv8 = ConvolutionLayer(256, 128)
        self.up4 = UpSample(128)
        self.conv9 = ConvolutionLayer(128, 64)
        self.predict = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        d1 = self.down1(x1)

        x2 = self.conv2(d1)
        d2 = self.down2(x2)

        x3 = self.conv3(d2)
        d3 = self.down3(x3)

        x4 = self.conv4(d3)
        d4 = self.down4(x4)

        x5 = self.conv5(d4)
        up1 = self.up1(x5)

        x6 = self.conv6(torch.cat((x4, up1), dim=1))
        up2 = self.up2(x6)

        x7 = self.conv7(torch.cat((x3, up2), dim=1))
        up3 = self.up3(x7)

        x8 = self.conv8(torch.cat((x2, up3), dim=1))
        up4 = self.up4(x8)

        x9 = self.conv9(torch.cat((x1, up4), dim=1))
        mask = self.predict(x9)
        return mask


if __name__ == '__main__':
    # l = ConvolutionLayer(3,64)
    # m = DownSample()
    unet = UNet(3, 1)
    x = torch.randn(1, 3, 512, 512)
    # print(l(x).shape)
    # print(m(x).shape)
    print(unet(x).shape)
