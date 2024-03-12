# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

main file

@author: tadahaya
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import argparse
import yaml

from tqdm.auto import tqdm

from src.arguments import get_args
from src.models import *
from src.utils import save_experiment, load_experiments, visualize_images, visualize_attention
from src.data_handler import get_dataset
from src.trainer import Trainer

# ToDo 240312
# 2つの実装を行う
# setup.pyを用いたパッケージ化
# CLIを用いた実行
# 直近やること
# parse_argsとconfigの統合

def make_optimizer(params, name, **kwargs):
    """ optimizerの作成 """
    return optim.__dict__[name](params, **kwargs)


def get_args():
    """ 引数の取得 """
    parser = argparse.ArgumentParser(description="Yaml file for training")
    parser.add_argument("--config", type=str, required=True, help="Yaml file for training")
    args.parser.parse_args()
    return args


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save_model_every", type=int, default=0)
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    # yamlの読み込み
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    # dataの読み込み
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    # モデル等の準備
    model = VitForClassification(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2) # AdamW使っている
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(config, model, optimizer, loss_fn, args.exp_name, device=device)
    trainer.train(
        trainloader, testloader, epochs, save_model_evry_n_epochs=save_model_every_n_epochs
        )









if __name__ == "__main__":
    args = get_args() # ToDo argsを準備する
    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f"Log file has been saved to {completed_log_dir}")