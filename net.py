import torch
import torch.nn as nn
import torch.nn.functional as F

from block import BFLinear, BFConv2d
from functions import BFConf

class BFSimpleNet(nn.Module):
    def __init__(self, bf_conf, num_classes=10, cuda=True):
        super(BFSimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = BFConv2d(16, 32, 3, padding=0,
            bf_conf=BFConf(bf_conf["conv2"]), bias=False, cuda=cuda)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = BFConv2d(32, 64, 3, padding=0,
            bf_conf=BFConf(bf_conf["conv3"]), bias=False, cuda=cuda)
        self.bn3 = nn.BatchNorm2d(64)
        # Setting Only Convolution as BFConv
        # self.fc1 = BFLinear(64 * 5 * 5, 1024,
        #     bf_conf=BFConf(bf_conf["fc1"]), cuda=cuda)
        # self.fc2 = BFLinear(1024, 1024,
        #     bf_conf=BFConf(bf_conf["fc2"]), cuda=cuda)
        self.fc1 = nn.Linear(64*5*5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        self.bf_conf = bf_conf
        self.cuda = cuda


    def forward(self, x):
                                        #  3x32x32
        x = F.relu(self.conv1(x))       # 16x32x32
        x = self.bn1(x)
        x = self.pool1(x)               # 16x16x16
        x = F.relu(self.conv2(x))       # 32x14x14
        x = self.bn2(x)
        x = self.pool2(x)               # 32x 7x 7
        x = F.relu(self.conv3(x))       # 64x 5x 5
        x = self.bn3(x)
        x = x.view(-1, 64 * 5 * 5)      # 1600
        x = F.relu(self.fc1(x))         # 1024
        x = F.relu(self.fc2(x))         # 1024

        x = self.fc3(x)                 # 10
        return x


class SimpleNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0, bias=False) # changed to bfconv
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=0, bias=False) # changed to bfconv
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)


    def forward(self, x):
                                        #  3x32x32
        x = F.relu(self.conv1(x))       # 16x32x32
        x = self.bn1(x)
        x = self.pool1(x)               # 16x16x16
        x = F.relu(self.conv2(x))       # 32x14x14
        x = self.bn2(x)
        x = self.pool2(x)               # 32x 7x 7
        x = F.relu(self.conv3(x))       # 64x 5x 5
        x = self.bn3(x)
        x = x.view(-1, 64 * 5 * 5)      # 1600
        x = F.relu(self.fc1(x))         # 1024
        x = F.relu(self.fc2(x))         # 1024
        x = self.fc3(x)                 # 10
        return x
