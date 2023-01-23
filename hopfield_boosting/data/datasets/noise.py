import torch

class GaussianNoise(torch.utils.data.Dataset):
    def __init__(self, len, img_size, mean, std):
        self.len = len
        self.img_size = list(img_size)
        self.mean = torch.tensor(mean).reshape(-1, 1, 1)
        self.std = torch.tensor(std).reshape(-1, 1, 1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        generator = torch.Generator()
        generator.manual_seed(idx)
        return torch.randn(self.img_size, generator=generator) * self.std + self.mean, torch.empty((1,))


class UniformNoise(torch.utils.data.Dataset):
    def __init__(self, len, img_size):
        self.len = len
        self.img_size = list(img_size)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        generator = torch.Generator()
        generator.manual_seed(idx)
        return torch.rand(self.img_size, generator=generator), torch.empty((1,))
