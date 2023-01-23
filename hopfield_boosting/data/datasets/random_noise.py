import torch
import numpy as np
from PIL import Image

class RandomNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size=(32, 32), no_classes=10, transform=None, length=10000):
        self.no_classes = no_classes
        self.size = size
        self.transform = transform
        self.length = length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise ValueError('Illegal Index')
        rand = torch.rand(1, *self.size)
        if self.transform:
            rand = self.transform(Image.fromarray(np.array(rand[0].detach())))
        return rand, torch.randint(high=self.no_classes, size=[])


    def __len__(self):
        return self.length
