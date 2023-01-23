from contextlib import contextmanager
from itertools import count

import torch
from torch import nn
import numpy as np


def logmeanexp(beta, tensor, dim, ignore_negative_inf=False, keepdim=False):
    n = torch.tensor(tensor.size(dim))
    if ignore_negative_inf:
        no_neg_inf = torch.sum((torch.isinf(tensor) & (tensor < 0)).to(torch.int), dim=dim)
        n = n - no_neg_inf
    lse = 1/beta * torch.logsumexp(beta * tensor, dim=dim, keepdim=keepdim)
    return lse - 1 / beta * torch.log(n)


@contextmanager
def infer_loader(loader: torch.utils.data.DataLoader, model, device='cpu', max_samples=None):
    try:
        xs = []

        if max_samples:
            r = range(int(np.ceil(max_samples / loader.batch_size)))
        else:
            r = count(0)

        with torch.no_grad():
            for _, (x, y) in zip(r, loader):
                x = x.to(device)
                x = model(x)
                xs.append(x)

            x = torch.concat(xs, dim=0)

        yield x

    finally:
        del x

class Negative(nn.Module):
    def forward(self, x):
        return -x
    
class FirstElement(nn.Module):
    def forward(self, x):
        return x[0]