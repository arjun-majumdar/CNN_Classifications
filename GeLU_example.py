

import numpy as np
import matplotlib.pyplot as plt


"""
Implementing Gaussian Error Linear Unit (GeLU)  activation
function using sigmoid approximation.


Reference-
https://www.youtube.com/watch?v=kMpptn-6jaw
"""


def tanh_act_fn(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def sigmoid_act_fn(x):
    return 1 / (1 + np.exp(-x))


def gelu_fn(x, use_tanh = False):
    if use_tanh:
        return tanh_act_fn(x = x)
    else:
        return x * sigmoid_act_fn(x = 1.72 * x)


# Define input-
z = np.arange(-4, 4, 1e-2)

# Sanity check-
# gelu_fn(x = z, use_tanh = False).min(), gelu_fn(x = z, use_tanh = False).max()
# (-0.1618959531420815, 3.985830896288157)

# gelu_fn(x = z, use_tanh = True).min(), gelu_fn(x = z, use_tanh = True).max()
# (-0.17003944483437977, 3.9899263281297217)


# Visualize GeLU activation function-
plt.plot(z, gelu_fn(x = z, use_tanh = False), label = 'sigmoid approx')
plt.plot(z, gelu_fn(x = z, use_tanh = True), label = 'tanh approx')
plt.title("GeLU activation using different approximations")
plt.xlabel("input x")
plt.ylabel("GeLU(x)")
plt.legend(loc  = 'best')
plt.show()

