

import numpy as np
import matplotlib.pyplot as plt
import os, pickle
from tqdm import tqdm, trange
# from LARC import LARC
from lars import LARS

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


print(f"torch version: {torch.__version__}")

# Check if there are multiple devices (i.e., GPU cards)-
print(f"Number of GPU(s) available = {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch does not have access to GPU")

# Device configuration-
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device is {device}')


# Specify hyper-parameters
batch_size = 4096
num_classes = 10
num_epochs = 85


# MNIST dataset statistics:
# mean = tensor([0.1307]) & std dev = tensor([0.3081])
mean = np.array([0.1307])
std_dev = np.array([0.3081])

transforms_apply = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std_dev)
    ])


# MNIST dataset-
train_dataset = torchvision.datasets.MNIST(
        root = '/home/amajumdar/Downloads/.data/', train = True,
        transform = transforms_apply, download = True
        )

test_dataset = torchvision.datasets.MNIST(
        root = '/home/amajumdar/Downloads/.data/', train = False,
        transform = transforms_apply
        )

# Create dataloader-
train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size = batch_size,
        shuffle = True
        )

test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, batch_size = batch_size,
        shuffle = False
        )

# len(train_dataset), len(test_dataset)

# len(train_dataset), len(test_dataset)
num_steps_epoch = int(np.ceil(len(train_dataset) / batch_size))




class ResNet_Block(nn.Module):
    '''
    ResNet-18/34 block
    '''
    def __init__(
        self, num_inp_channels,
        num_channels, stride = 1,
        dropout = 0.2, use_1x1_conv = False
    ):
        super(ResNet_Block, self).__init__()

        self.num_inp_channels = num_inp_channels
        self.num_channels = num_channels
        self.stride = stride
        self.dropout = dropout
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
        self.dropout = nn.Dropout(p = self.dropout)

        if self.use_1x1_conv:
            self.conv3 = nn.Conv2d(
            in_channels = self.num_inp_channels, out_channels = num_channels,
            kernel_size = 1, padding = 0,
            stride = self.stride, bias = False
            )
            self.bn3 = nn.BatchNorm2d(num_features = self.num_channels)


    def forward(self, x):
        y = F.leaky_relu(self.bn1(self.conv1(x)))
        y = self.dropout(F.leaky_relu(self.bn2(self.conv2(y))))

        if self.use_1x1_conv:
            x = self.bn3(self.conv3(x))

        y += x
        return F.leaky_relu(self.dropout(y))


    def shape_computation(self, x):
        print(f"Input shape: {x.shape}")
        y = (self.bn1(self.conv1(x)))
        print(f"First conv layer output shape: {y.shape}")
        y = self.bn2(self.conv2(y))
        print(f"Second conv layer output shape: {y.shape}")

        if self.use_1x1_conv:
            x = self.bn3(self.conv3(x))
            print(f"Downsample with S = 2; identity connection output shape: {x.shape}")

        y += x
        print(f"Residual block output shape: {y.shape}")
        return None


def initialize_weights(model):
    for m in model.modules():
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


class CosineScheduler:
    def __init__(
        self, max_update,
        base_lr = 0.01, final_lr = 0,
        warmup_steps = 0, warmup_begin_lr = 0
    ):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps


    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase


    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + np.cos(
                np.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr




class ConvResNet(nn.Module):
    # Custom ResNet-18/34 variant for MNIST dataset.
    def __init__(self):
        super(ConvResNet, self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels = 1, out_channels = 32,
                kernel_size = 3, padding = 1,
                stride = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(num_features = 32)

        # Downsample-
        self.resblock1 = ResNet_Block(
            num_inp_channels = 32, num_channels = 32,
            stride = 2, dropout = 0.2,
            use_1x1_conv = True
        )

        self.resblock2 = ResNet_Block(
            num_inp_channels = 32, num_channels = 64,
            stride = 2, dropout = 0.2,
            use_1x1_conv = True
        )

        '''
        self.resblock3 = ResNet_Block(
            num_inp_channels = 64, num_channels = 64,
            stride = 2, dropout = 0.2,
            use_1x1_conv = True
        )
        '''

        self.avg_pool = nn.AvgPool2d(kernel_size = 3, stride = 2)
        self.op_layer = nn.Linear(
            in_features = 576, out_features = 10,
            bias = True
        )


    def forward(self, x):
        bs = x.size(0)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        # x = self.resblock3(x)
        x = self.avg_pool(x)
        x = x.flatten().view(bs, -1)
        x = self.op_layer(x)
        return x




# Initialize model and initialize parameters-
model = ConvResNet()
model.apply(initialize_weights)

# Save randomly initialized model-
torch.save(model.state_dict(), "ConvResNet_new_random_model.pth")


def count_trainable_params(model):
    # Count number of layer-wise parameters and total parameters-
    tot_params = 0
    for param in model.parameters():
        # print(f"layer.shape = {param.shape} has {param.nelement()} parameters")
        tot_params += param.nelement()

    return tot_params

print(f"\nConvResNet CNN has {count_trainable_params(model)} params\n")
# ConvResNet CNN has 83498 params


def exclude_from_wt_decay(
    named_params,
    weight_decay = 5e-5,
    skip_list = ['bias', 'bn']
    ):
    '''
    We are going to look through all of the model
    parameters (encoder + projection), and then,
    for bias and batch-norm, do NOT apply weight
    decay to them. Why?
    Think of y=mx+c case, where the bias 'c' controls
    the line shift above/below the origin, and so,
    we don't want to constraint that using L2
    regularization. Similarly, for batch-norm,
    the beta and gamma params control scale and shift
    and you don't want to restrict the scale and
    shift params using L2 regularization.
    '''

    # Do NOT apply weight decay to batch-norm or bias

    # params for which weight decay is to be applied-
    params = []

    # params for which weight decay is NOT to be applied-
    excluded_params = []

    skip_list = ['bias', 'bn']

    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            # print(f"excluded param: {name}, shape: {param.size()}")
            excluded_params.append(param)
        else:
            # print(f"param: {name}, shape: {param.size()}")
            params.append(param)

    return [
        {'params': params, 'weight_decay': weight_decay},
        {'params': excluded_params, 'weight_decay': 0.0}
    ]


model_params_wt_decay = exclude_from_wt_decay(
    named_params = model.named_parameters(),
    weight_decay = 5e-5,
    skip_list = ['bias', 'bn']
)


# Define loss function and LARS + SGD optimizer-
loss = nn.CrossEntropyLoss()

# LARS implementation is from pl_bolts.
optimizer = LARS(
    params = model_params_wt_decay,
    lr = 0.0,
    momentum = 0.0,
    weight_decay = 0.0,
    trust_coefficient = 0.001,
)


'''
LR Scheduler:

1. linear warmup (from lr = 0.0001) over 10 epochs (until lr = 0.03)
2. cosine decay without restarts until 80th epoch (lr = 0.0001)
'''
# Decay lr in cosine manner unitl 80th epoch-
scheduler = CosineScheduler(
    max_update = 80 * num_steps_epoch, base_lr = 0.03 * 16,
    final_lr = 0.0001, warmup_steps = 10 * num_steps_epoch,
    warmup_begin_lr = 0.0001
)

step = 1


def train_model_progress(model, train_loader, train_dataset):
    '''
    Function to perform one epoch of training by using 'train_loader'.
    Returns loss and number of correct predictions for this epoch.
    '''
    running_loss = 0.0
    running_corrects = 0.0

    model.train()

    with tqdm(train_loader, unit = 'batch') as tepoch:
        for images, labels in tepoch:
            tepoch.set_description(f"Training: ")

            images = images.to(device)
            labels = labels.to(device)

            # Get model predictions-
            outputs = model(images)

            # Compute loss-
            J = loss(outputs, labels)

            # Empty accumulated gradients-
            optimizer.zero_grad()

            # Perform backprop-
            J.backward()

            # Update parameters-
            optimizer.step()

            # Update LR scheduler & LARS params-
            global step
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler(step)
            step += 1

            # Compute model's performance statistics-
            running_loss += J.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            running_corrects += torch.sum(predicted == labels.data)

            tepoch.set_postfix(
                loss = running_loss / len(train_dataset),
                accuracy = (running_corrects.double().cpu().numpy() / len(train_dataset)) * 100
            )


    train_loss = running_loss / len(train_dataset)
    train_acc = (running_corrects.double() / len(train_dataset)) * 100


    # return running_loss, running_corrects
    return train_loss, train_acc.cpu().numpy()


def test_model_progress(model, test_loader, test_dataset):
    total = 0.0
    correct = 0.0
    running_loss_val = 0.0

    with torch.no_grad():
        with tqdm(test_loader, unit = 'batch') as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Testing: ")

                images = images.to(device)
                labels = labels.to(device)

                # Set model to evaluation mode-
                model.eval()

                # Predict using trained model-
                outputs = model(images)
                _, y_pred = torch.max(outputs, 1)

                # Compute validation loss-
                J_val = loss(outputs, labels)

                running_loss_val += J_val.item() * labels.size(0)

                # Total number of labels-
                total += labels.size(0)

                # Total number of correct predictions-
                correct += (y_pred == labels).sum()

                tepoch.set_postfix(
                    test_loss = running_loss_val / len(test_dataset),
                    test_acc = 100 * (correct.cpu().numpy() / total)
                )


    # return (running_loss_val, correct, total)
    val_loss = running_loss_val / len(test_dataset)
    val_acc = (correct / total) * 100

    return val_loss, val_acc.cpu().numpy()


# Python3 dict to contain training metrics-
train_history = {}

# Parameter to track 'best' test accuracy metric-
best_test_acc = 90


# TRAIN CNN loop-
for epoch in range(1, num_epochs + 1):

    # Update LR scheduler & LARS params-
    for param_group in optimizer.param_groups:
        param_group['lr'] = scheduler(epoch)
    # optimizer.param_groups[0]['lr'] = scheduler(epoch = epoch)

    # Train and validate model for 1 epoch-
    train_loss, train_acc = train_model_progress(
        model, train_loader,
        train_dataset
    )

    test_loss, test_acc = test_model_progress(
        model, test_loader,
        test_dataset
    )

    print(f"\nepoch: {epoch} train loss = {train_loss:.4f}, "
          f"train accuracy = {train_acc:.2f}%, test loss = {test_loss:.4f}"
          f", test accuracy = {test_acc:.2f}% & "
          f"LR = {optimizer.param_groups[0]['lr']:.7f}\n")

    train_history[epoch] = {
        'loss': train_loss, 'acc': train_acc,
        'test_loss': test_loss, 'test_acc': test_acc,
        'lr': optimizer.param_groups[0]['lr']
    }

    # Save best weights achieved until now-
    if (test_acc > best_test_acc):
        # update best test acc highest acc-
        best_test_acc = test_acc

        print(f"Saving model with highest test acc = {test_acc:.2f}%\n")

        # Save trained model with 'best' validation accuracy-
        torch.save(model.state_dict(), "ConvResNet_LARS_wtdecay_best_model.pth")


# Save training metrics as Python3 dict for later analysis-
with open("ConvResNet_LARS_wtdecay_MNIST_train_history.pkl", "wb") as file:
    pickle.dump(train_history, file)
del file


"""
# Traning Visualizations:

# plt.figure(figsize = (9, 7))
plt.plot(list(train_history.keys()), [train_history[k]['acc'] for k in train_history.keys()], label = 'train acc')
plt.plot(list(train_history.keys()), [train_history[k]['test_acc'] for k in train_history.keys()], label = 'test acc')
plt.title("ConvResNet: Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy (%)")
plt.legend(loc = 'best')
plt.show()

plt.plot(list(train_history.keys()), [train_history[k]['loss'] for k in train_history.keys()], label = 'train loss')
plt.plot(list(train_history.keys()), [train_history[k]['test_loss'] for k in train_history.keys()], label = 'test loss')
plt.title("ConvResNet: Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc = 'best')
plt.show()

plt.plot(list(train_history.keys()), [train_history[k]['lr'] for k in train_history.keys()])
plt.xlabel("epochs")
plt.ylabel("lr")
plt.title("ConvResNet: Learning-Rate")
plt.show()




# Load trained CNN model-
trained_model = ConvResNet()
trained_model.load_state_dict(torch.load("ConvResNet_best_model.pth"))

# Get test metrics of 'best' trained model-
test_loss, test_acc = test_model_progress(
    model = trained_model, test_loader = test_loader,
    test_dataset = test_dataset
)

print("ConvResNet 'best' trained testing metrics: ",
      f"loss = {test_loss:.4f} & acc = {test_acc:.2f}%"
)
# ConvResNet 'best' trained testing metrics:  loss = 0.0167 & acc = 99.53%
"""

