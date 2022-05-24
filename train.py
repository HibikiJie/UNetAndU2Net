from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from u_net import UNet
from u2net import U2Net
from dataset import DriveDataset
import os


class Trainer:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")  # 设置设备
        self.net = UNet(3, 1).to(self.device)  # 实例U-Net
        if os.path.exists('unet.pth'):  # 加载权重，如果存在的话
            self.net.load_state_dict(torch.load('unet.pth', map_location='cpu'))
        self.dataset = DriveDataset()  # 实例数据集
        self.data_loader = DataLoader(self.dataset, 3, True, drop_last=True)  # 实例数据加载器
        self.loss_func = nn.MSELoss()  # 实例二值交叉熵
        self.optimizer = torch.optim.Adam(self.net.parameters())  # 实例adam优化器

    def train(self):  # 训练
        for epoch in range(100000):  # 迭代epoch
            for i, (image, target) in enumerate(self.data_loader):
                image = image.to(self.device)
                target = target.to(self.device)

                out = self.net(image)  # 预测
                loss = self.loss_func(out, target)  # 计算损失

                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 优化
                print(epoch, loss.item())

            if epoch % 1 == 0:
                torch.save(self.net.state_dict(),'unet.pth')
                save_image([image[0], target[0].expand(3, 256, 256), out[0].expand(3, 256, 256)], f'{epoch}.jpg',normalize=True,range=(0,1))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()


