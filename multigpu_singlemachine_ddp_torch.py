

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from ResNet18_swish_torch import ResNet18, init_weights


"""
EXPERIMENTAL: Multi-GPU, Single Machine DDP PyTorch Training

ResNet-18 CNN + CIFAR-10 dataset


Refer-
https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
"""


def setup(rank: int, world_size: int) -> None:
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	dist.init_process_group(
		"nccl", rank = rank,
		world_size = world_size
	)


def cleanup():
    dist.destroy_process_group()


def prepare_cifar10_dataset(
		rank: int, world_size: int,
		batch_size = 256, pin_memory = False,
		num_workers = 0, path_dataset = "/home/majumdar/Downloads/.data/"
	) -> DataLoader:
	
	# Define transformations for training and test sets-
    transform_train = transforms.Compose(
        [
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.0305, 0.0296, 0.0342)),
        ]
    )

    transform_test = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4942, 0.4846, 0.4498),
            std = (0.0304, 0.0295, 0.0342)),
        ]
    )
	
    train_dataset = torchvision.datasets.CIFAR10(
		root = path_dataset, train = True,
		download = True, transform = transform_train
		)

    test_dataset = torchvision.datasets.CIFAR10(
        root = path_dataset, train = False,
        download = True, transform = transform_test
    )
	
    train_sampler = DistributedSampler(
		dataset = train_dataset, num_replicas = world_size,
		rank = rank, shuffle = False,
		drop_last = False
    )
	
    test_sampler = DistributedSampler(
		dataset = test_dataset, num_replicas = world_size,
		rank = rank, shuffle = False,
		drop_last = False
    )
	
    train_loader = DataLoader(
		dataset = train_dataset, batch_size = batch_size,
		pin_memory = pin_memory, num_workers = num_workers,
		drop_last = False, 	shuffle = False,
		sampler = train_sampler
    )
	
    test_loader = DataLoader(
		dataset = test_dataset, batch_size = batch_size,
		pin_memory = pin_memory, num_workers = num_workers,
		drop_last = False, 	shuffle = False,
		sampler = test_sampler
    )
	
    # return train_loader, test_loader, train_dataset, test_dataset, train_sampler, test_sampler
    return train_loader, test_loader, train_dataset, test_dataset


def main(
        rank: int, world_size: int,
        num_epochs = 50
    ):
    # setup process groups-
    setup(rank, world_size)

	# prepare CIFAR-10 dataloader- 
    train_loader, test_loader, train_dataset, test_dataset = prepare_cifar10_dataset(
		rank = rank, world_size = world_size,
		batch_size = 256, pin_memory = False,
		num_workers = 0, path_dataset = "/home/majumdar/Downloads/.data/"
	)

    # Instantiate model and move to correct device-
    model = ResNet18(beta = 1.0).to(rank)
	
    # Apply weights initialization-
    model.apply(init_weights)
	
	# Wrap model with DDP device_ids tell DDP where is your model
	# output_device tells DDP where to output, in our case, it is rank
	# find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(
		module = model, device_ids = [rank],
		output_device = rank, find_unused_parameters = True
	)
    '''
    Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass.
    This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your
    model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a
    false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
    '''


    # TRAIN LOOP (for n epochs):
    
    # Define loss function and optimizer-
    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
         params = model.parameters(), lr = 0.001,
         momentum = 0.9, weight_decay = 5e-4
         )
    

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        running_corrects = 0.0
    
        model.train()

        # Inform DistributedSampler about current epoch-
        train_loader.sampler.set_epoch(epoch)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(rank)
            labels = labels.to(rank)

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

            '''                
            global step
            optimizer.param_groups[0]['lr'] = custom_lr_scheduler.get_lr(step)
            step += 1
            ''' 
            
            # Compute model's performance statistics-
            running_loss += J.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            running_corrects += torch.sum(predicted == labels.data)
            
        train_loss = running_loss / len(train_dataset)
        train_acc = (running_corrects.double() / len(train_dataset)) * 100
        print(f"GPU: {rank}, epoch = {epoch}; train loss = {train_loss:.4f} & train accuracy = {train_acc:.2f}%")


    # For now, save model at last epoch-
    # if self.gpu_id == 0 and epoch % self.save_every == 0:
    if rank == 0:
        ckp = model.module.state_dict()
        PATH = "ResNet18_lastepoch.pth"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    cleanup()




if __name__ == '__main__':
    world_size = torch.cuda.device_count()

    mp.spawn(
         fn = main,
         args = (world_size, 50),
         nprocs = world_size
    )

    # CUDA_VISIBLE_DEVICES=0,1,2 python multigpu_ddp_torch.py

    
