'''AlexNet model modified for CIFAR-10 in PyTorch.
from https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py
Edited some layer configurations to match blocking
'''
import torch.nn as nn

from functions import SetConv2dLayer

NUM_CLASSES = 10

class AlexNet(nn.Module):
    def __init__(self, bf_conf, num_classes=NUM_CLASSES, bwg_boost=1.0):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            SetConv2dLayer("conv1", bf_conf, 3, 64, 3, stride=2, padding=1, bwg_boost=bwg_boost),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(64, 192, kernel_size=3, padding=1),
            SetConv2dLayer("conv2", bf_conf, 64, 192, 3, padding=1, bwg_boost=bwg_boost),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            SetConv2dLayer("conv3", bf_conf, 192, 384, 3, padding=1, bwg_boost=bwg_boost),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            SetConv2dLayer("conv4", bf_conf, 384, 256, 3, padding=1, bwg_boost=bwg_boost),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            SetConv2dLayer("conv5", bf_conf, 256, 256, 3, padding=1, bwg_boost=bwg_boost),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x