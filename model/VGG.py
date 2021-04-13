'''VGG11/13/16/19 in Pytorch.
Implemented by kuangliu, https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
Edited some layer configurations to match blocking
'''
import torch
import torch.nn as nn

from functions import SetConv2dLayer

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, bf_conf, out_channels, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(bf_conf, cfg[vgg_name])
        self.classifier = nn.Linear(512, out_channels)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, bf_conf, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                        # nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        SetConv2dLayer(str(i), bf_conf, in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(bf_conf, out_channels):
    return VGG(bf_conf, out_channels, 'VGG11')

def VGG13(bf_conf, out_channels):
    return VGG(bf_conf, out_channels, 'VGG13')

def VGG16(bf_conf, out_channels):
    return VGG(bf_conf, out_channels, 'VGG16')

def VGG19(bf_conf, out_channels):
    return VGG(bf_conf, out_channels, 'VGG19')
