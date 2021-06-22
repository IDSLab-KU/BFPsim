'''DenseNet model modified for CIFAR-10 in PyTorch.
Implemented by kuangliu, https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
Edited some layer configurations to match blocking
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import SetConv2dLayer


class Bottleneck(nn.Module):
    def __init__(self, bf_conf, in_planes, growth_rate, bwg_boost=1.0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.conv1 = SetConv2dLayer("conv1", bf_conf, in_planes, 4*growth_rate, kernel_size=1, bias=False, bwg_boost=bwg_boost)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        # self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.conv2 = SetConv2dLayer("conv2", bf_conf, 4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False, bwg_boost=bwg_boost)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, bf_conf, name, in_planes, out_planes, bwg_boost=1.0):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.conv = SetConv2dLayer(name, bf_conf, in_planes, out_planes, kernel_size=1, bias=False, bwg_boost=bwg_boost)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, bf_conf, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, bwg_boost=1.0):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        # self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        self.conv1 = SetConv2dLayer("conv1", bf_conf, 3, num_planes, kernel_size=3, padding=1, bias=False, bwg_boost=bwg_boost)

        self.dense1 = self._make_dense_layers(bf_conf, "dense1", block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(bf_conf, "trans1", num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(bf_conf, "dense2", block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(bf_conf, "trans2", num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(bf_conf, "dense3", block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(bf_conf, "trans3", num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(bf_conf, "dense4", block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, bf_conf, name, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            if name in bf_conf and str(i) in bf_conf[name]:
                layers.append(block(bf_conf[name][str(i)], in_planes, self.growth_rate))
            else:
                empty_dict = dict()
                layers.append(block(empty_dict, in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121(bf_conf, out_channels, bwg_boost=1.0):
    return DenseNet(bf_conf, Bottleneck, [6,12,24,16], growth_rate=32, num_classes=out_channels, bwg_boost=bwg_boost)

def DenseNet169(bf_conf, out_channels, bwg_boost=1.0):
    return DenseNet(bf_conf, Bottleneck, [6,12,32,32], growth_rate=32, num_classes=out_channels, bwg_boost=bwg_boost)

def DenseNet201(bf_conf, out_channels, bwg_boost=1.0):
    return DenseNet(bf_conf, Bottleneck, [6,12,48,32], growth_rate=32, num_classes=out_channels, bwg_boost=bwg_boost)

def DenseNet161(bf_conf, out_channels, bwg_boost=1.0):
    return DenseNet(bf_conf, Bottleneck, [6,12,36,24], growth_rate=48, num_classes=out_channels, bwg_boost=bwg_boost)

def DenseNetCifar(bf_conf, out_channels, bwg_boost=1.0):
    return DenseNet(bf_conf, Bottleneck, [6,12,24,16], growth_rate=12, num_classes=out_channels, bwg_boost=bwg_boost)
