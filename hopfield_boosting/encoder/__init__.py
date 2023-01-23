import torch
from torch import nn

class CNNOODWrapper(nn.Module):
    def __init__(self, cnn, preprocess) -> None:
        super(CNNOODWrapper, self).__init__()
        self.preprocess = preprocess
        self.module = cnn

    def forward(self, x):
        assert x.shape[-2:] == (32, 32)  # ensure CIFAR-10 size
        with torch.no_grad():
            x = self.preprocess(x)
        x = self.module(x)
        return x
