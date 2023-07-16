

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from cifar10_dataloader import get_cifar10_data


"""
Inception (GoogleNet)

Inception module has 7 layers. These modules are stacked on top of
each other. To limit the computational requiremenets, they use 1x1
conv layers for dimensionality reduction modules - allows to
increase depth and width of the CNN.

1. Inception module, naive version-
previous layer -> 1x1 conv, 3x3 conv, 5x5 conv and 3x3 max pooling ->
filter concatenation

2. Inception module with dimension reduction-
reduce feature map before feeding to 3x3 or 5x5 conv layer. Refer
to paper for figure.

They apply dimensionality reduction and projections, and its inspired
from embeddings.

Overall, there are 9 Inception modules. At the output of some of the
Inception modules, they have used an auxilliary classifier - it
computes loss (already) at some points in the CNN. The authors were
concerned about the gradient backprop. So, by adding these auxilliary
classifiers, they hope to improve this gradient backprop flow and
therefore, improve vanishing gradients problem. It also seems to
provide additional regularization.
During training, their (auxilliary classifiers) loss gets added to the
total loss of the network with a discounted weight (0.3).

They use AveragePooling layer vs. dense layer.


Refer-
https://www.youtube.com/watch?v=r92siBwTI8U
"""


class InceptionBlock(nn.Module):
    def __init__(
        self, inp_channels,
        feature_maps
        ):
        super().__init__()

        # 1x1 projection 1-
        self.projection1 = nn.Sequential(
            nn.Conv2d(
                in_channels = inp_channels, out_channels = feature_maps[0],
                kernel_size = 1, padding = 0,
                stride = 1, bias = False,
                # groups = 3
            ),
            nn.BatchNorm2d(num_features = feature_maps[0]),
            nn.ReLU()
            # nn.BatchNorm2d(num_features = feature_maps[0])
            )

        # projection 2-
        self.projection2 = nn.Sequential(
            nn.Conv2d(
                in_channels = inp_channels, out_channels = feature_maps[1],
                kernel_size = 1, padding = 0,
                stride = 1, bias = False,
            ),
            nn.BatchNorm2d(num_features = feature_maps[1]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = feature_maps[1], out_channels = feature_maps[2],
                kernel_size = 3, padding = 1,
                stride = 1, bias = False,
            ),
            nn.BatchNorm2d(num_features = feature_maps[2]),
            nn.ReLU()
        )

        # projection 3-
        self.projection3 = nn.Sequential(
            nn.Conv2d(
                in_channels = inp_channels, out_channels = feature_maps[1],
                kernel_size = 1, padding = 0,
                stride = 1, bias = False,
            ),
            nn.BatchNorm2d(num_features = feature_maps[1]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = feature_maps[1], out_channels = feature_maps[2],
                kernel_size = 5, padding = 2,
                stride = 1, bias = False,
            ),
            nn.BatchNorm2d(num_features = feature_maps[2]),
            nn.ReLU()
        )

        # projection 4-
        self.projection4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size = 3, stride = 1,
                padding = 1
            ),
            nn.Conv2d(
                in_channels = inp_channels, out_channels = feature_maps[5],
                kernel_size = 1, padding = 0,
                stride = 1, bias = False,
            ),
            nn.BatchNorm2d(num_features = feature_maps[5]),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.projection1(x)
        out2 = self.projection2(x)
        out3 = self.projection3(x)
        out4 = self.projection4(x)

        return torch.cat([out1, out2, out3, out4], axis = 1)


class InceptionNet(nn.Module):
    def __init__(
        self, inp_channels,
        feat_map
        ):
        '''
        Modified for CIFAR-10 dataset.
        '''
        super().__init__()

        self.inp_channels = inp_channels
        self.feat_map = feat_map

        self.conv1 = nn.Conv2d(
            in_channels = 3, out_channels = 64,
            kernel_size = 3, padding = 1,
            stride = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(num_features = 64)

        # Define inception blocks-
        self.inception_block = nn.ModuleList(
            InceptionBlock(
                inp_channels = self.inp_channels[i], feature_maps = self.feat_map[i]) \
                for i in range(len(self.feat_map))
            )

        # type(inception_block), len(inception_block)
        # (torch.nn.modules.container.ModuleList, 9)

        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        # avg_pool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.avg_pool = nn.AvgPool2d(kernel_size = 3, stride = 2)

        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                '''
                # Do not initialize bias (due to batchnorm)-
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                '''
            elif isinstance(m, nn.BatchNorm2d):
                # Standard initialization for batch normalization-
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))

        for i, incp_b in enumerate(self.inception_block):
            x = incp_b(x)
            if i == 2 or i == 5 or i == 7:
                x = self.pool(x)
            # print(f"i = {i}, out shape: {x.size()}")

        return self.avg_pool(x).squeeze()



'''
# Define feature map for inception blocks-
feat_map = [
    [64, 96, 128, 16, 32, 32],
    [128, 128, 192, 32, 96, 64],
    [192, 96, 208, 16, 48, 64],
    [160, 112, 224, 24, 64, 64],
    [128, 128, 256, 24, 64, 64],
    [112, 144, 288, 32, 64, 64],
    [256, 160, 320, 32, 128, 128],
    [256, 160, 320, 32, 128, 128],
    [384, 192, 384, 48, 128, 128]
    ]

# inp_channels = [192, 256, 480, 512, 512, 512, 528, 832, 1024]
inp_channels = [64, 352, 576, 672, 672, 704, 752, 1024, 1024]

model = InceptionNet(
    inp_channels = inp_channels, feat_map = feat_map
    )
'''

