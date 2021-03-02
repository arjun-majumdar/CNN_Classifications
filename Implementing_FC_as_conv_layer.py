

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


"""
Example code to implement Fully Connected (FC) layer(s) as Convolutional
layer(s)

For use in 'Convolutional Implementation of Sliding Windows Object Detection'
algorithm.
"""


# Define random input-
x = torch.rand((5, 3, 14, 14))
# (batch size, channel, height, width)

x.shape
# torch.Size([5, 3, 14, 14])


# Define a convolutional layer-
c1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding = 0, bias = True)

# Output of a conv layer (O) = ((W - K + 2P) / S) + 1

# Pass the input through the conv layer-
x_c1 = c1(x)

x_c1.shape
# torch.Size([5, 16, 10, 10])


# Define a max pooling layer-
max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

# Output of a max pool layer (W') = ((W - f) / S) + 1

# Pass the input volume through the max pooling layer-
x_pool = max_pool(x_c1)

x_pool.shape
# torch.Size([5, 16, 5, 5])


'''
The first FC layer normally has 400 neurons in it. It is now being implemented
as a convolutional layer using:
    1.) filter size = 5 x 5
    2.) stride = 1
    3.) padding = 0
    4.) number of filters = 400
'''
fc_as_c2 = nn.Conv2d(in_channels=16, out_channels=400, kernel_size=5, stride=1, padding=0, bias=True)

x_c2 = fc_as_c2(x_pool)

x_c2.shape
# torch.Size([5, 400, 1, 1])


'''
Similarly, the second FC layer normally having 400 neurons in it is also being
implemented as a convolutional layer using:
    1.) filter size = 1 x 1
    2.) stride = 1
    3.) padding = 0
    4.) number of filters = 400
'''
fc_as_c3 = nn.Conv2d(in_channels=400, out_channels=400, kernel_size=1, stride=1, padding=0, bias = True)

x_c3 = fc_as_c3(x_c2)

x_c3.shape
# torch.Size([5, 400, 1, 1])


'''
Finally, the output layer normally having 4 neurons in it is being implemented
as a convolutional layer using:
    1.) filter size = 1 x 1
    2.) stride = 1
    3.) padding = 0
    4.) number of filters = 4
'''
op_as_c4 = nn.Conv2d(in_channels=400, out_channels=4, kernel_size=1, stride=1, padding=0, bias = True)

x_op = op_as_c4(x_c3)

x_op.shape
# torch.Size([5, 4, 1, 1])


# To view the output of the first image as computed by the output layer as a
# conv layer-
x_op.detach().numpy()[0]
'''
array([[[ 0.07920169]],

       [[ 0.06151699]],

       [[-0.01613115]],

       [[-0.08001155]]], dtype=float32)
'''

# Reshape into a vector of dimension = 4
x_op.detach().numpy()[0].reshape(4)
# array([ 0.07920169,  0.06151699, -0.01613115, -0.08001155], dtype=float32)

# Output the index having the largest value for the third image-
np.argmax(x_op.detach().numpy()[2])
# 0


'''
NOTE:

While defining the conv net, softmax function is NOT included since it's
included in the loss function.

The cross-entropy loss applies softmax function for us.

loss = nn.CrossEntropyLoss()    # applies softmax for us!
'''


