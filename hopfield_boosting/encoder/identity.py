from torch import nn


class IdentityEncoder(nn.Module):
    def forward(self, x):
        return x
