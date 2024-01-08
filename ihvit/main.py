# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

main file

@author: tadahaya
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from tqdm.auto import tqdm

from src.augmentations import get_aug
from src.arguments import get_args
from src.models import *
from src.utils import AverageMeter, knn_monitor, Logger, file_exist_check
from src.data_handler import get_dataset
from src.optimizers import get_optimizer, LR_Scheduler


def main(device, args):
    train_loader = DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs
        )
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    # get_dataset --> Dataset
    memory_loader = DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs
        )
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs
        )
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model).to(device)
    # ToDo model読み込みの作成　→　modelsにbackboneを作る







if __name__ == "__main__":
    args = get_args() # ToDo argsを準備する
    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f"Log file has been saved to {completed_log_dir}")