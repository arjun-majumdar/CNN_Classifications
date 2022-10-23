

import torch
import torch.nn as nn
import numpy as np


"""
Custom Activation Function in PyTorch

Reference-
https://towardsdatascience.com/extending-pytorch-with-custom-activation-functions-2d8b065ef2fa
"""


def silu(x):
    '''
    Sigmoid Linear Unit (SiLU) activation function.
    '''
    return x * torch.sigmoid(x)


class SiLU_act(nn.Module):
    '''
    Class to apply SiLU activation function, element-wise.
    '''
    def __init__(self):
        super(SiLU_act, self).__init__()


    def forward(self, x):
        return silu(x)


# Initialize activation function-
activation_fn = SiLU_act()


# Initialize a neural network architecture-
model = nn.Sequential(
        nn.Linear(in_features = 100, out_features = 20),
        activation_fn,
        nn.Linear(in_features = 20, out_features = 10),
        nn.Softmax(dim = 1)
)


class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()

        self.layer1 = nn.Linear(in_features = 100, out_features = 50)
        self.layer2 = nn.Linear(in_features = 50, out_features = 20)
        self.layer3 = nn.Linear(in_features = 20, out_features = 10)


    def forward(self, x):
        x = silu(self.layer1(x))
        x = silu(self.layer2(x))
        x = nn.Softmax(dim = 1)(self.layer3(x))
        return x


# Random input-
x = torch.randn(32, 100)

# Initialize an instance of model-
model = Sample()

# Ger (random) prediction-
pred = model(x)

pred.shape, x.shape
# (torch.Size([32, 10]), torch.Size([32, 100]))



