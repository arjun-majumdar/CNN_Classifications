

import torch
import torch.nn as nn


"""
Swish activation function where 'beta' parameter is trained.
"""


class Swish(nn.Module):
    def __init__(self, beta = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta, requires_grad = True))

    def forward(self, x):
        return x * torch.sigmoid(x * self.beta)

