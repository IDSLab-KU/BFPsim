'''ResNet model modified for CIFAR-10 in PyTorch.
Implemented by kuangliu, https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
Edited some layer configurations to match blocking

Disclaimer : layer with kernel size is not 3 will lead unexpected result.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import SetConv2dLayer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, bf_conf, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d( in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = SetConv2dLayer("conv1", bf_conf, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = SetConv2dLayer("conv2", bf_conf, planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                SetConv2dLayer("shortcut", bf_conf, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Resnet 18 and 34 Doesn't have Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = SetConv2dLayer("conv2", bf_conf, planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)   
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BFResNet(nn.Module):
    def __init__(self, block, num_blocks, bf_conf, num_classes):
        super(BFResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0],
            bf_conf=bf_conf["layer1"], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1],
            bf_conf=bf_conf["layer2"], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2],
            bf_conf=bf_conf["layer3"], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3],
            bf_conf=bf_conf["layer4"], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, bf_conf, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            layers.append(block(self.in_planes, planes, bf_conf[str(i)], strides[i]))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, bf_conf, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = SetConv2dLayer("conv1", bf_conf, 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, bf_conf, "layer1", 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, bf_conf, "layer2", 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, bf_conf, "layer3", 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, bf_conf, "layer4", 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, bf_conf, name, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            if name in bf_conf and str(i) in bf_conf[name]: # Not normal layer
                layers.append(block(bf_conf[name][str(i)], self.in_planes, planes, strides[i]))
            else: # Normal Layer
                empty_dict = dict()
                layers.append(block(empty_dict, self.in_planes, planes, strides[i]))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(bf_conf, num_classes):
    return ResNet(BasicBlock, bf_conf, [2, 2, 2, 2], num_classes=num_classes)

def ResNet34(bf_conf, num_classes):
    return ResNet(BasicBlock, bf_conf, [3, 4, 6, 3], num_classes=num_classes)

def ResNet50(bf_conf, num_classes):
    return ResNet(Bottleneck, bf_conf, [3, 4, 6, 3], num_classes=num_classes)

def ResNet101(bf_conf, num_classes):
    return ResNet(Bottleneck, bf_conf, [3, 4, 23, 3], num_classes=num_classes)

def ResNet152(bf_conf, num_classes):
    return ResNet(Bottleneck, bf_conf, [3, 8, 36, 3], num_classes=num_classes)
