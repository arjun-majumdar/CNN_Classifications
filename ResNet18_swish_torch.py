

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
ResNet-18/34 using Swish activation function with trainable beta hyper-parameter.
"""


class ResNet_Block(nn.Module):
    '''
    Residual block for ResNet-18/34.
    '''
    def __init__(
        self, num_inp_channels,
        num_channels, beta,
        stride = 1, use_1x1_conv = False, 
    ):
        super(ResNet_Block, self).__init__()
        
        # Trainable parameter for swish activation function-
        # self.beta = nn.Parameter(torch.tensor(beta, requires_grad = True))
        
        self.num_inp_channels = num_inp_channels
        self.num_channels = num_channels
        self.stride = stride
        self.use_1x1_conv = use_1x1_conv
        self.beta = beta
    
        self.conv1 = nn.Conv2d(
            in_channels = self.num_inp_channels, out_channels = self.num_channels,
            kernel_size = 3, padding = 1,
            stride = self.stride, bias = False
        )
        self.bn1 = nn.BatchNorm2d(num_features = self.num_channels)
        
        self.conv2 = nn.Conv2d(
            in_channels = self.num_channels, out_channels = self.num_channels,
            kernel_size = 3, padding = 1,
            stride = 1, bias = False
        )
        self.bn2 = nn.BatchNorm2d(num_features = self.num_channels)
        
        if self.use_1x1_conv:
            self.conv3 = nn.Conv2d(
            in_channels = self.num_inp_channels, out_channels = num_channels,
            kernel_size = 1, padding = 0,
            stride = self.stride, bias = False
            )
            self.bn3 = nn.BatchNorm2d(num_features = self.num_channels)
    

    def swish_fn(self, x):
        return x * torch.sigmoid(x * self.beta)    
    
    
    def forward(self, x):
        # y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn1(self.conv1(x))
        y = self.swish_fn(x = y)
        # y = self.dropout(F.relu(self.bn2(self.conv2(y))))
        y = self.bn2(self.conv2(y))
        y = self.swish_fn(x = y)
        
        if self.use_1x1_conv:
            x = self.bn3(self.conv3(x))
            
        y += x
        # return F.relu(self.dropout(y))
        return self.swish_fn(x = y)
    
    
    def shape_computation(self, x):
        print(f"Input shape: {x.shape}")
        y = F.relu(self.bn1(self.conv1(x)))
        print(f"First conv layer output shape: {y.shape}")
        y = self.bn2(self.conv2(y))
        print(f"Second conv layer output shape: {y.shape}")
        
        if self.use_1x1_conv:
            x = self.bn3(self.conv3(x))
            print(f"Downsample with S = 2; identity connection output shape: {x.shape}")
            
        y += x
        print(f"Residual block output shape: {y.shape}")
        return None


class ResNet18(nn.Module):
    def __init__(self, beta = 1.0):
        super(ResNet18, self).__init__()
        
        # Trainable parameter for swish activation function-
        self.beta = nn.Parameter(torch.tensor(beta, requires_grad = True))
        
        self.conv1 = nn.Conv2d(
            in_channels = 3, out_channels = 64,
            kernel_size = 3, padding = 1,
            stride = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(num_features = 64)
        
        self.resblock1 = ResNet_Block(
            num_inp_channels = 64, num_channels = 64,
            stride = 1, use_1x1_conv = False,
            beta = self.beta
        )
        
        self.resblock2 = ResNet_Block(
            num_inp_channels = 64, num_channels = 64,
            stride = 1, use_1x1_conv = False,
            beta = self.beta
        )
        
        # Downsample-
        self.resblock3 = ResNet_Block(
            num_inp_channels = 64, num_channels = 128,
            stride = 2, use_1x1_conv = True,
            beta = self.beta
        )
        
        self.resblock4 = ResNet_Block(
            num_inp_channels = 128, num_channels = 128,
            stride = 1, use_1x1_conv = False,
            beta = self.beta
        )

        # Downsample-
        self.resblock5 = ResNet_Block(
            num_inp_channels = 128, num_channels = 256,
            stride = 2, use_1x1_conv = True,
            beta = self.beta
        )

        self.resblock6 = ResNet_Block(
            num_inp_channels = 256, num_channels = 256,
            stride = 1, use_1x1_conv = False,
            beta = self.beta
        )

        # Downsample-
        self.resblock7 = ResNet_Block(
            num_inp_channels = 256, num_channels = 512,
            stride = 2, use_1x1_conv = True,
            beta = self.beta
        )

        self.resblock8 = ResNet_Block(
            num_inp_channels = 512, num_channels = 512,
            stride = 1, use_1x1_conv = False,
            beta = self.beta
        )
        
        self.avg_pool = nn.AvgPool2d(kernel_size = 3, stride = 2)


    def swish_fn(self, x):
        return x * torch.sigmoid(x * self.beta)    
    
    
    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn1(self.conv1(x))
        x = self.swish_fn(x = x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.avg_pool(x).squeeze()
        return x


@torch.no_grad()
def init_weights(m):
    # print(m)
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.fill_(1.0)
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.fill_(1.0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.fill_(1.0)
        if m.bias is not None:
            m.bias.fill_(1.0)

    return None


