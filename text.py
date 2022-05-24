from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from u_net import UNet
from u2net import U2Net
from dataset import DriveDataset
import os
import cv2
import numpy


class Explorer:

    def __init__(self):
        self.net = UNet(3, 1)
        self.net.load_state_dict(torch.load('unet.pth'))
        self.net.eval()

    def explore(self, image):
        image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255
        out = self.net(image).squeeze(0).permute(1, 2, 0) * 255
        out = out.detach().numpy().astype(numpy.uint8)
        return out


if __name__ == '__main__':
    explorer = Explorer()
    image = cv2.imread('data/test/images/01_test.tif')
    h, w, c = image.shape
    print(image.shape)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = explorer.explore(image)
    print(image)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    # video = cv2.VideoCapture('D:/data/chapter8/DRIVE/test/mask/01_test_mask.gif')
    # _, mask = video.read()
    # image = numpy.where(mask[:, :, 0] > 125, image, 0)
    print(image.shape)
    cv2.imshow('image', image.astype(numpy.uint8))
    cv2.waitKey()
