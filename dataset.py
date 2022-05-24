import torch
import random
import cv2
from torch.utils.data import Dataset


class DriveDataset(Dataset):

    def __init__(self,root='data/training'):
        super(DriveDataset, self).__init__()
        self.dataset = []
        start = 20
        for i in range(1, 21):
            image_path = f'{root}/images/{i+start}_training.tif'
            label_path = f'{root}/1st_manual/{i + start}_manual1.gif'
            self.dataset.append((image_path, label_path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, label_path = self.dataset[item]
        image = cv2.imread(image_path)
        video = cv2.VideoCapture(label_path)
        _, mask_label = video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_label = cv2.cvtColor(mask_label, cv2.COLOR_BGR2GRAY)
        h, w = mask_label.shape
        w = random.randint(0, w-256)
        h = random.randint(0, h-256)
        image = image[h:h+256, w:w+256]
        mask_label = mask_label[h:h + 256, w:w + 256]
        image = torch.from_numpy(image).float().permute(2, 0, 1)/255
        mask_label = torch.from_numpy(mask_label).unsqueeze(0).float()/255
        return image, mask_label

if __name__ == '__main__':
    pass