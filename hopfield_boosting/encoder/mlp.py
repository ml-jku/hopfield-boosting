from torch import nn


class MLP(nn.Module):
    def __init__(self, no_layers, in_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.no_layers = no_layers
        assert self.no_layers >= 2, 'MLP requires at least two layers'

        self.in_layer = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        self.hidden = nn.Sequential(*(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(no_layers - 2)))

        self.model = nn.Sequential(self.in_layer, self.hidden, self.out_layer)

    def forward(self, x):
        return self.model(x)
