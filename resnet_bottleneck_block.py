

import torch
import torch.nn as nn
import torch.nn.functional as F
from swish_act_fn import Swish


class resnet_block(nn.Module):
    '''
    Residual bottleneck block for ResNet-50/101/152, etc.
    '''
    def __init__(
        self, inp_channels,
        output_channels, swish_fn,
        downsample = False
        ):
        super().__init__()

        self.skip_connection = None
        self.downsample = downsample
        self.swish_fn = swish_fn

        if self.downsample:
            self.conv_l = nn.Conv2d(
                in_channels = inp_channels, out_channels = output_channels,
                kernel_size = 1, stride = 2,
                padding = 0, bias = False
                )
            self.bn1 = nn.BatchNorm2d(num_features = output_channels)
        else:
            self.conv_l = nn.Conv2d(
                in_channels = inp_channels, out_channels = output_channels,
                kernel_size = 1, stride = 1,
                padding = 0, bias = False
            )
            self.bn1 = nn.BatchNorm2d(num_features = output_channels)

        self.conv_l2 = nn.Conv2d(
            in_channels = output_channels, out_channels = output_channels,
            kernel_size = 3, stride = 1,
            padding = 1, bias = False
        )
        self.bn2 = nn.BatchNorm2d(num_features = output_channels)

        self.conv_l3 = nn.Conv2d(
            in_channels = output_channels, out_channels = output_channels * 4,
            kernel_size = 1, stride = 1,
            padding = 0, bias = False
        )
        self.bn3 = nn.BatchNorm2d(num_features = output_channels * 4)

        if self.downsample:
            self.skip_connection = nn.Conv2d(
                in_channels = inp_channels, out_channels = output_channels * 4,
                kernel_size = 1, stride = 2,
                padding = 0, bias = False
                )
            self.skip_bn = nn.BatchNorm2d(num_features = output_channels * 4)
        elif inp_channels != output_channels * 4:
            self.skip_connection = nn.Conv2d(
                    in_channels = inp_channels, out_channels = output_channels * 4,
                    kernel_size= 1, stride = 1,
                    padding = 0, bias = False
                    )
            self.skip_bn = nn.BatchNorm2d(num_features = output_channels * 4)

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
        # y = F.leaky_relu(self.bn1(self.conv_l(x)))
        y = self.swish_fn(self.bn1(self.conv_l(x)))
        y = self.swish_fn(self.bn2(self.conv_l2(y)))
        y = self.bn3(self.conv_l3(y))

        # Use skip/identity connection to match channels-
        if self.downsample:
            x = self.skip_bn(self.skip_connection(x))
        elif self.skip_connection is not None and self.downsample is not True:
            x = self.skip_bn(self.skip_connection(x))

        y += x
        return self.swish_fn(y)


class ResNet50(nn.Module):
    def __init__(self, beta = 1.0):
        super().__init__()

        self.swish_fn = Swish(beta = beta)
        
        self.conv_inp = nn.Conv2d(
            in_channels = 3, out_channels = 64,
            kernel_size = 3, padding = 1,
            stride = 1, bias = False
        )
        self.bn_inp = nn.BatchNorm2d(num_features = 64)

        # Define residual blocks-
        self.resblock1 = resnet_block(
                inp_channels = 64, output_channels = 64,
                downsample = False, swish_fn = self.swish_fn
                )

        self.resblock2 = resnet_block(
                inp_channels = 256, output_channels = 64,
                downsample = False, swish_fn = self.swish_fn
                )

        self.resblock3 = resnet_block(
                inp_channels = 256, output_channels = 64,
                downsample = False, swish_fn = self.swish_fn
                )

        # Downsample spatial dimensions-
        self.resblock4 = resnet_block(
                inp_channels = 256, output_channels = 128,
                downsample = True, swish_fn = self.swish_fn
                )

        self.resblock5 = resnet_block(
                inp_channels = 512, output_channels = 128,
                downsample = False, swish_fn = self.swish_fn
                )

        self.resblock6 = resnet_block(
                inp_channels = 512, output_channels = 128,
                downsample = False, swish_fn = self.swish_fn
                )

        self.resblock7 = resnet_block(
                inp_channels = 512, output_channels = 128,
                downsample = False, swish_fn = self.swish_fn
                )

        # Downsample spatial dimensions-
        self.resblock8 = resnet_block(
                inp_channels = 512, output_channels = 256,
                downsample = True, swish_fn = self.swish_fn
                )

        self.resblock9 = resnet_block(
                inp_channels = 1024, output_channels = 256,
                downsample = False, swish_fn = self.swish_fn
                )

        self.resblock10 = resnet_block(
                inp_channels = 1024, output_channels = 256,
                downsample = False, swish_fn = self.swish_fn
                )

        self.resblock11 = resnet_block(
                inp_channels = 1024, output_channels = 256,
                downsample = False, swish_fn = self.swish_fn
                )

        self.resblock12 = resnet_block(
                inp_channels = 1024, output_channels = 256,
                downsample = False, swish_fn = self.swish_fn
                )

        self.resblock13 = resnet_block(
                inp_channels = 1024, output_channels = 256,
                downsample = False, swish_fn = self.swish_fn
                )

        # Downsample spatial dimensions-
        self.resblock14 = resnet_block(
                inp_channels = 1024, output_channels = 512,
                downsample = True, swish_fn = self.swish_fn
                )

        self.resblock15 = resnet_block(
                inp_channels = 2048, output_channels = 512,
                downsample = False, swish_fn = self.swish_fn
                )

        self.resblock16 = resnet_block(
                inp_channels = 2048, output_channels = 512,
                downsample = False, swish_fn = self.swish_fn
                )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.output_layer = nn.Linear(in_features = 2048, out_features = 10, bias = True)


    def forward(self, x):
        x = F.leaky_relu(self.bn_inp(self.conv_inp(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
        x = self.resblock14(x)
        x = self.resblock15(x)
        x = self.avgpool(x)
        # x = x.view(-1, 2048)
        x = x.squeeze()
        return self.output_layer(x)

