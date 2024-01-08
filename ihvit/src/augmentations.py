# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

augmentations

@author: tadahaya
"""
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np


imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]] # mean and std of ImageNet


def get_aug(name, image_size, train, train_classifier):
    if train:
        augmentation = SimSiamTransform(image_size)
    elif train == False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    return augmentation


class SimSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = 224 if image_size is None else image_size
        p_blur = 0.5 if image_size > 32 else 0
        # comments:
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), 
            T.RandomGrayScale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


def to_pil_image(pic, mode=None):
    """
    convert a tensor or an ndarray to PIL Image
    
    pic: tensor or ndarray
    mode: color space and pixel depth

    """
    if not(isinstance(pic, torch.Tensor)) or isinstance(pic, np.ndarray):
        raise TypeError("!! pic should be Tensor or ndarray !!")
    elif isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional')
        elif pic.ndimension() == 2:
            # if 2D, add channel dimension
            pic = pic.unsqueeze(0)
    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional')
        elif pic.ndim == 2:
            pic = np.expand_dims(pic, 2)
    
    npimg = pic
    if isinstance(pic, torch.Tensor):
        if pic.is_floating_point() and mode != 'F':
            pic = pic.mul(255).byte()
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic should be Tensor or ndarray')

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dytpe == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError('Incorrect mode')
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError('Incorrect mode')
        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'
    
    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError('Incorrect mode')
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'

    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError('Incorrect mode')
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'
    
    if mode is None:
        raise TypeError('Check input type')

    return Image.fromarray(npimg, mode=mode)


class Transform_single():
    def __init__(self, image_size, train, normalize=imagenet_mean_std):
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), interpolation=Image.BICUBIC
                    ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(*normalize)
            ])
        else:
            self.transform = T.Compose([
                T.Resize(int(image_size) * (8/7), interpolation=Image.BICUBIC), # 224 -> 256
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(*normalize)
            ])
        
    def __call__(self):
        pass
