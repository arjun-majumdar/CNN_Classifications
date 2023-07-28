

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np


class ResNet_Block(nn.Module):
    '''
    Residual block for ResNet-18/34.
    '''
    def __init__(
        self, num_inp_channels,
        num_channels, stride = 1,
        use_1x1_conv = False,
    ):
        super().__init__()

        # Trainable parameter for swish activation function-
        # self.beta = nn.Parameter(torch.tensor(beta, requires_grad = True))

        self.num_inp_channels = num_inp_channels
        self.num_channels = num_channels
        self.stride = stride
        self.use_1x1_conv = use_1x1_conv

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


    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))

        if self.use_1x1_conv:
            x = self.bn3(self.conv3(x))

        # y += x
        y = y + x
        return F.relu(y)


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


class ResUNet(nn.Module):
    def __init__(
        self, inp_channels = 3,
        op_channels = 2, features = [64, 128, 256, 512],
    ):
        '''
        Replace double conv block with a ResNet-18 block. The resulting architecure is
        called Residual U-Net.
        '''
        super().__init__()

        self.contraction = nn.ModuleList()
        self.expansion = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.contraction = nn.ModuleList()
        self.expansion = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.bottleneck = ResNet_Block(
            num_inp_channels = features[-1], num_channels = features[-1] * 2, use_1x1_conv = True
            )

        # final conv output layer-
        self.final_conv = nn.Conv2d(
            in_channels = features[0], out_channels = op_channels,
            kernel_size = 1, padding = 0
        )

        # Encoder/Expansion network/path-
        for feature in features:
            self.contraction.append(
                ResNet_Block(
                    num_inp_channels = inp_channels, num_channels = feature, use_1x1_conv = True
                    )
                )
            inp_channels = feature

        # Decoder/Expansion network/path-
        for feature in reversed(features):
            self.expansion.append(
                nn.ConvTranspose2d(
                    in_channels = feature * 2, out_channels = feature,
                    kernel_size = 2, stride = 2,
                    padding = 0
                    )
                )
            self.expansion.append(
                ResNet_Block(num_inp_channels = feature * 2, num_channels = feature, use_1x1_conv = True
                             )
                )
        
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
        # Python3 list for skip connections between contraction and expansion levels-
        skip_connections = []

        for layer in self.contraction:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.expansion), 2):
            # Do conv transpose 2d (upsampling)-
            x = self.expansion[idx](x)

            # Get relevant residual connection-
            skip_connection = skip_connections[idx // 2]

            # When dimensions mismatch-
            if x.shape != skip_connection.shape:
                x = TF.resize(x, antialias = True, size = skip_connection.shape[2:])

            # Concatenate (batch, channels, height, width) along channel dimension-
            concat_skip = torch.cat((skip_connection, x), dim = 1)

            # Feed to residual block-
            x = self.expansion[idx + 1](concat_skip)

        return self.final_conv(x)

