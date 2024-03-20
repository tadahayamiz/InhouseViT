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

from .src.arguments import get_args
from .src.models import *
from .src.utils import save_experiment, load_experiments, visualize_images, visualize_attention
from .src.trainer import Trainer
from .src.data_handler import prepare_data


def get_args():
    """ 引数の取得 """
    parser = argparse.ArgumentParser(description="Yaml file for training")
    parser.add_argument("--config_path", type=str, required=True, help="Yaml file for training")
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    # argsの取得
    args = get_args()
    # yamlの読み込み
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["config_path"] = args.config_path
    config["exp_name"] = args.exp_name
    # dataの読み込み
    trainloader, testloader, _ = prepare_data(batch_size=config["batch_size"])
    # モデル等の準備
    model = VitForClassification(config)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2) # AdamW使っている
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(config, model, optimizer, loss_fn, args.exp_name, device=config["device"])
    trainer.train(
        trainloader, testloader, save_model_evry_n_epochs=config["save_model_every"]
        )


if __name__ == "__main__":
    main()