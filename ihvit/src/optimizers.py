# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

optimizers

@author: tadahaya
"""

import torch
import numpy as np

def get_optimizer(name, model, lr, momentum, weight_decay):
    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name':'base',
        'params':[param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr':lr
    }, {
        'name':'predictor',
        'params':[param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr':lr
    }]
    if name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer
    

class LR_Scheduler(object):
    def __init__(
            self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr,
            final_lr, iter_per_epoch, constant_predictor_lr=False
            ):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
                # なぞ
        self.iter += 1
        self.current_lr = lr
        return lr
    
    def get_lr(self):
        return self.current_lr
    

if __name__ == "__main__":
    # 可視化用？
    import torchvision
    model = torchvision.models.resnet50() # 変更
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    epochs = 100
    n_iter = 1000
    scheduler = LR_Scheduler(optimizer, 10, 1, epochs, 3, 0, n_iter)
    import matplotlib.pyplot as plt
    lrs = []
    for epoch in range(epochs):
        for it in range(n_iter):
            lr = scheduler.step()
            lrs.append(lr)
    plt.plot(lrs)
    plt.show()