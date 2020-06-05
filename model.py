import torch
import numpy as np
from torch import nn
from torch.nn import functional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(torch.nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        if seed:
            torch.manual_seed(0)

        super(QNetwork, self).__init__()

        # self state input, 64 output
        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = functional.relu(self.lin1(state))
        x = functional.relu(self.lin2(x))
        x = self.lin3(x)

        return x
