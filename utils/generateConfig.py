
import torch
import torch.optim as optim
import torch.nn as nn

def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

def GenerateConfig(args):
    print("------------------------------------")
    for name, module in args.net.named_modules():
        if isinstance(module, nn.Linear):
            print("LINEAR: " + name)
        if isinstance(module, nn.Conv2d):
            print("CONV2D: " + name)
    # named_layers = dict(args.net.named_modules())
    # print(named_layers)