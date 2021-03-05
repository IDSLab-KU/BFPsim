import torch
import torch.nn as nn
import torch.nn.functional as F

from block import BFLinear, BFConv2d
from functions import BFConf

class BFSimpleNet(nn.Module):
    def __init__(self, bf_conf, num_classes=10, cuda=True):
        super(BFSimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = BFConv2d(16, 32, 3, padding=0,
            bf_conf=BFConf(bf_conf["conv2"]), bias=False, cuda=cuda)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = BFConv2d(32, 64, 3, padding=0,
            bf_conf=BFConf(bf_conf["conv3"]), bias=False, cuda=cuda)

        self.fc1 = BFLinear(64 * 5 * 5, 1024,
            bf_conf=BFConf(bf_conf["fc1"]), cuda=cuda)
        self.fc2 = BFLinear(1024, 1024,
            bf_conf=BFConf(bf_conf["fc2"]), cuda=cuda)
        self.fc3 = nn.Linear(1024, num_classes)


    def forward(self, x):
                                        #  3x32x32
        x = F.relu(self.conv1(x))       # 16x32x32
        x = self.pool1(x)               # 16x16x16
        x = F.relu(self.conv2(x))       # 32x14x14
        x = self.pool2(x)               # 32x 7x 7
        x = F.relu(self.conv3(x))       # 64x 5x 5
        x = x.view(-1, 64 * 5 * 5)      # 1600
        x = F.relu(self.fc1(x))         # 1024
        x = F.relu(self.fc2(x))         # 1024
        x = self.fc3(x)                 # 10
        return x


class SimpleNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0, bias=False) # changed to bfconv
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=0, bias=False) # changed to bfconv

        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)


    def forward(self, x):
                                        #  3x32x32
        x = F.relu(self.conv1(x))       # 16x32x32
        x = self.pool1(x)               # 16x16x16
        x = F.relu(self.conv2(x))       # 32x14x14
        x = self.pool2(x)               # 32x 7x 7
        x = F.relu(self.conv3(x))       # 64x 5x 5
        x = x.view(-1, 64 * 5 * 5)      # 1600
        x = F.relu(self.fc1(x))         # 1024
        x = F.relu(self.fc2(x))         # 1024
        x = self.fc3(x)                 # 10
        return x


# Another option
# https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

# Resnet Implkementation from
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# Told it has 93% Accuracy
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BFBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bf_conf, stride):
        super(BFBasicBlock, self).__init__()
        self.conv1 = BFConv2d(in_planes, planes, 
            kernel_size=3, bf_conf=BFConf(bf_conf["conv1"]), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BFConv2d(planes, planes,
            kernel_size=3, bf_conf=BFConf(bf_conf["conv2"]), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BFConv2d(in_planes, self.expansion*planes,
                    kernel_size=1, bf_conf=BFConf(bf_conf["shortcut"]), stride=stride, bias=False),
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, bf_conf, num_classes):
        super(ResNet, self).__init__()
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


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def BFResNet18(bf_conf, num_classes):
    return ResNet(BFBasicBlock, [2, 2, 2, 2], bf_conf, num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])