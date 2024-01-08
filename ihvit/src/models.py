# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

models file

@author: tadahaya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50 # ここは要変更か

def get_model():
    return model = SimSiam()


def D(p, z, version:str="simplified"):
    """
    differenceの算出方法を決定する
    zがencodeした後に勾配を止める側, pがprojectionまでしてそのまま流す側
    
    """
    if version == "original":
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p * z).sum(dim=1).mean() # cosine similarityになっている
    elif version == "simplified":
        # 速いバージョンらしい, おそらくCコンパイル済みのFを使っているからか
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception
        # なんもなしのExcpetionとは珍しい気も, KeyErrorとかでいいんじゃ


class projection_MLP(nn.Module):
    """
    projection用のMLP, 小文字始まりのclassは気になる
    baseline用とのこと
    以下のコメントがあった, fに実装されているものだと不都合だったか？
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.    
    
    """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True) # inplaceの使い分けは相変わらず気になる
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            # nn.BatchNorm1d(hidden_dim)
            nn.BatchNorm1d(out_dim)
            # repositoryではここがout_dimではなくhiddenになっているが, defaultが同じだから成立している
        )
        self.num_layers = 3 # hardになっているが初期値みたいなもの

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x) # 最終はReLUなし
        else:
            raise Exception
            # 先ほどと同様。コーダーの癖か
        return x


class prediction_MLP(nn.Module):
    """ bottleneck structureとされている """
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        """
        baselineとされている
        論文の内容を引用しているので見ながらやった方がわかりやすいかも

        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        # BNをlayer2に入れると学習が不安定になる模様, 本文参照

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    

class SimSiam(nn.Module):
    """ backboneをnn.Moduleで与える必要あり """
    def __init__(self, backbone=resnet50()):
        super().__init__()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor # 変数として呼び出す意義？
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2 # 互い違いに平均
        return {"loss":L} # lossはdictで返している
    

# ToDo
# SimSiam classのデフォルトで与えているresnet50を何らかより軽量なものへ置換


# backbone
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=False, dilation=dilation
        )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
                 ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width64')
        if dilation > 1:
            raise NotImplementedError('dilation > 1 not supported in BasicBlock')
        # both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(
            self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None
                 ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self, block, layers, num_classes=10, zero_init_residual=False,
            groups=1, width_per_group=64, replace_stride_with_dilation=None,
            norm_layer=None
                 ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element is the tuple indicates if we should replace 
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("!! Check the length of replace_stride_with_dilation !!")
        self.groups = groups
        self.base_width = width_per_group
        # CIFAR10 kernel 7 -> 3, stride 2 -> 1, padding 3 -> 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # コメントアウトされている, adaptiveの方が楽だからか
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init_constant_(m.bias, 0)
        # zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behave like an identity
        # This improves the model by 0.2~0.3%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            layers = []
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, 
                    self.base_width, previous_dilation, norm_layer
                    )
                )
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))
            return nn.Sequential(*layers)
    

        def _resnet(self):
            raise NotImplementedError
        
        # Next ここから

        def resnet18(self):
            raise NotImplementedError