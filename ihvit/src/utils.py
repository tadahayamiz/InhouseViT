# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

utils

@author: tadahaya
"""

import os
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter # tensorboardを使っている
from torch import Tensor
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/49921721/runtimeerror-main-thread-is-not-in-main-loop-with-matplotlib-and-flask
import matplotlib.pyplot as plt


# accuracy.py
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True) # 231212 outputが何のインスタンスか
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size)) # 231212 100単位に揃えている
        return res


# knn_monitor.py
# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, epoch, k=200, t=0.1, hide_progress=False):
    net.eval()
    classes = len(memory_data_loader.dataset.classes) # torchのdatasetでclassesが定義されている必要があるか
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0.0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc="Feature extracting", leave=False, disable=hide_progress):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1) # l2-normalize
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc="kNN", disable=hide_progress)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True) # 高速化か
            feature = net(data)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy':total_top1 / total_num * 100}) # percentage
    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature banck ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1) # ref torch.topk
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp() # 指数的に重み
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B * K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1
        )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


# average_meter.py
class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.log = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# if __name__ == "__main__":
#     meter = AverageMeter('sldk')
#     print(meter.log)

# file_exist_fn.py
def file_exist_check(file_dir):
    if os.path.isdir(file_dir):
        for i in range(2, 1000):
            if not os.path.isdir(file_dir + f"({i})"):
                file_dir += f"({i})"
                break
    return file_dir

# logger.py
class Logger(object):
    def __init__(self, log_dir, tensorboard=True, matplotlib=True):
        self.reset(log_dir, tensorboard, matplotlib)
    
    def reset(self, log_dir=None, tensorboard=True, matplotlib=True):
        if log_dir is not None:
            self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir) if tensorboard else None
        self.plotter = Plotter if matplotlib else None
        self.counter = OrderedDict()

    def update_scalers(self, ordered_dict):
        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1
            if self.writer:
                self.writer.add_scaler(key, value, self.counter[key])
        if self.plotter:
            self.plotter.update(ordered_dict)
            self.plotter.save(os.path.join(self.log_dir, "plotter.svg"))

class Plotter(object):
    def __init__(self):
        self.logger = OrderedDict()

    def update(self, ordered_dict):
        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.logger.get(key) is None:
                self.logger[key] = [value]
            else:
                self.logger[key].append(value)
        
    def save(self, file, **kwargs):
        fig, axes = plt.subplots(
            nrows=len(self.logger), ncols=1, figsize=(8, 2 * len(self.logger))
            )
        fig.tight_layout()
        for ax, (key, value) in zip(axes, self.logger.items()):
            ax.plot(value)
            ax.set_title(key)
        plt.savefig(file, **kwargs)
        plt.close()