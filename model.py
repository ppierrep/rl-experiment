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

    
class DuelingQNetwork(torch.nn.Module):
    """Dueling deep Q network"""

    def __init__(self, state_size, action_size, seed):
        super(DuelingQNetwork, self).__init__()

        # self state input, 64 output
        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(64, 64)
        
        self.linValue = nn.Linear(64, 128)
        self.linAvantage = nn.Linear(64, 128)
        
        # State value
        self.val = nn.Linear(128, 1)

        # Advantage for each action
        self.adv = nn.Linear(128, action_size)

    def forward(self, state):
        x = functional.relu(self.lin1(state))
        x = functional.relu(self.lin2(x))
        adv = self.linAvantage(x)
        adv = self.adv(adv)

        val = self.linValue(x)
        val = self.val(val)
        
        adv_mean = torch.mean(adv, dim=1, keepdim=True)

        x = val + adv - adv_mean

        return x
