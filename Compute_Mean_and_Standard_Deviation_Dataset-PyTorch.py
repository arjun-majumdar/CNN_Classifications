

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader


"""
PyTorch: COmpute mean and standard deviation of dataset.

Refer-
https://www.youtube.com/watch?v=y6IEcEBRZks
"""


# Load CIFAR-10 dataset-
train_dataset = datasets.CIFAR10(
    root = 'data/', train = True,
    transform = transforms.ToTensor(), download = True
    )

test_dataset = datasets.CIFAR10(
    root = 'data/', train = False,
    transform = transforms.ToTensor(), download = True
    )

train_loader = DataLoader(
    dataset = train_dataset, batch_size = 256,
    shuffle = True
    )

test_loader = DataLoader(
    dataset = test_dataset, batch_size = 256,
    shuffle = True
    )



def calculate_mean_stddev(data_loader):
    '''
    Compute mean and standard-deviation across all channels for the input
    data loader.
    '''
    # VAR(X) = E(X^2) - E(X) ^ 2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in data_loader:
        channels_sum += torch.mean(data, dim = [0, 2, 3])
        # We don't want mean across channels (1st dimension), hence it is ignored.
        
        channels_squared_sum += torch.mean(data ** 2, dim = [0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std_dev = (channels_squared_sum / num_batches - (mean ** 2)) * 0.5
    # You cannot sum the standard deviation as it is not a linear operation.
    
    return mean, std_dev


mean_train, std_dev_train = calculate_mean_stddev(data_loader = train_loader)
mean_test, std_dev_test = calculate_mean_stddev(data_loader = test_loader)

print(f"CIFAR-10 train dataset: mean = {mean_train} & std-dev = {std_dev_train}")
# CIFAR-10 train dataset: mean = tensor([0.4914, 0.4821, 0.4465]) & std-dev = tensor([0.0305, 0.0296, 0.0342])

print(f"CIFAR-10 train dataset: mean = {mean_test} & std-dev = {std_dev_test}")
# CIFAR-10 train dataset: mean = tensor([0.4942, 0.4846, 0.4498]) & std-dev = tensor([0.0304, 0.0295, 0.0342])

