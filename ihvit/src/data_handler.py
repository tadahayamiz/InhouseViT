# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler

@author: tadahaya
"""
import torch
from torch.utils.data import Dataset
from torchvision import datasets

# datasets.__init__.py
def get_dataset(
        dataset, data_dir, transform, train=True, download=False, debug_subset_size=None
        ):
    if dataset == 'mnist':
        dataset = datasets.MNIST(
            data_dir, train=train, transform=transform, download=download
            )
    elif dataset == "stl10":
        dataset = datasets.STL10(
            data_dir, train=train, transform=transform, download=download
        )
    elif dataset == "cifar10":
        dataset = datasets.CIFAR10(
            data_dir, tarin=train, transform=transform, download=download
        )
    elif dataset == "cifar100":
        dataset = datasets.CIFAR100(
            data_dir, train=train, transform=transform, download=download
        )
    elif dataset == "imagenet":
        dataset = datasets.ImageNet(
            data_dir, split="train" if train == True else "val",
            transform=transform, download=download
        )
    elif dataset == "random":
        dataset = RandomDataset()
    else:
        raise NotImplementedError
    return dataset


# datasets.random_dataset.py
class RandomDataset(Dataset):
    def __init__(self, root=None, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.size = 1000 # hard??

    def __getitem__(self, idx):
        if idx < self.size:
            return [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))], [0, 0, 0]
        else:
            raise Exception

    def __len__(self):
        return self.size