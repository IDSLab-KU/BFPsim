'''MobileNet model modified for CIFAR-10 in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
Implemented by kuangliu, https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
Edited some layer configurations to match blocking
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import SetConv2dLayer

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, bf_conf, in_planes, out_planes, stride=1, bwg_boost=1.0):
        super(Block, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.conv1 = SetConv2dLayer("conv1", bf_conf, in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False, bwg_boost=self.bwg_boost)
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = SetConv2dLayer("conv2", bf_conf, in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, bwg_boost=self.bwg_boost)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetv1(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, bf_conf, num_classes=10, bwg_boost=1.0):
        super(MobileNetv1, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bwg_boost = bwg_boost
        self.conv1 = SetConv2dLayer("conv1", bf_conf, 3, 32, kernel_size=3, stride=1, padding=1, bias=False, bwg_boost=self.bwg_boost)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(bf_conf, in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, bf_conf, in_planes):
        layers = []
        for i, x in enumerate(self.cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if str(i) in bf_conf:
                layers.append(Block(bf_conf[str(i)], in_planes, out_planes, stride, bwg_boost=self.bwg_boost))
            else:
                empty_dict = dict()
                layers.append(Block(empty_dict, in_planes, out_planes, stride, bwg_boost=self.bwg_boost))

            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
