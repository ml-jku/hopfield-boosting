import torch
import torchvision
import numpy as np

class KaMNIST(torch.utils.data.Dataset):
    def __init__(self, root='../datasets/ood_datasets/kamnist/', transform=None):
        self.kamnist_x = np.load(f'{root}/X_KaMNIST_ICLR.npy')
        self.kamnist_y = np.load(f'{root}/y_KaMNIST_ICLR.npy')
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(torchvision.transforms.ToPILImage()(self.kamnist_x[idx])), self.kamnist_y[idx]
    
    def __len__(self):
        return len(self.kamnist_x)
