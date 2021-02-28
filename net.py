import torch
import torch.nn as nn
import torch.nn.functional as F

from block import BFLinear, BFConv2d


class SimpleNet(nn.Module):
    def __init__(self, classifier = 10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = BFConv2d(16, 32, 3, padding=0) # changed to bfconv
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = BFConv2d(32, 64, 3, padding=0) # changed to bfconv

        self.fc1 = BFLinear(64 * 5 * 5, 1024)
        self.fc2 = BFLinear(1024, 1024)
        self.fc3 = nn.Linear(1024, classifier)


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
    
