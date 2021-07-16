import torch
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from utils.logger import Log

def str2tuple(v):
    r = []
    v = v.replace(" ","").replace("(","").replace(")","").split(",")
    for i in v:
        r.append(int(i))
    return tuple(r)

def str2bool(v):
    if v.lower() in ["true", "t", "1"]: return True
    elif v.lower() in ["false", "f", "0"]: return False
    else: raise ValueError("str2bool: not parsable")

DIR_DICT = {
    "WI" :  0,
    "WO" :  1,
    "FX" : 10,
    "FY" : 11,
    "FC" : 12
}

def DictKey(d, v):
    for key in d:
        if d[key] == v:
            return key

def DirKey(v):
    for key in DIR_DICT:
        if DIR_DICT[key] == v:
            return key

def SaveModel(args, suffix):
    PATH = "%s_%s.model"%(args.save_prefix,suffix)
    Log.Print("Saving model file as %s"%PATH)
    torch.save(args.net.state_dict(), PATH)


from bfp.conf import BFPConf
from bfp.module import BFPLinear, BFPConv2d

def GetValueFromBFPConf(bfp_dict, attr_str):
    if attr_str in bfp_dict: # Layer configuration is found
        # If type is normal Conv2d
        if "type" in bfp_dict[attr_str] and bfp_dict[attr_str]["type"] == "torch.Conv2d":
            return None
        else:   # Found Config!
            return BFPConf(bfp_dict[attr_str])
    elif "default" in bfp_dict: # If default value is set, use the default value
        return BFPConf(bfp_dict["default"])
    else: # If no default value is set, don't replace
        return None

def ReturnBFPConv2d(ta, bfpc):
    if bfpc == None:
        return None
    bias = True if ta.bias != None else False
    new = BFPConv2d(in_channels=ta.in_channels, out_channels=ta.out_channels, kernel_size=ta.kernel_size, bfp_conf=bfpc, stride=ta.stride, padding=ta.padding, dilation=ta.dilation, groups=ta.groups, bias=bias, padding_mode=ta.padding_mode)
    return new

def ReplaceLayers(net, bfp_dict, name="net"):
    
    for attr_str in dir(net):
        ta = getattr(net, attr_str)
        # print(type(ta),end="\t")
        bfpc = GetValueFromBFPConf(bfp_dict, name+"."+attr_str)
        if type(ta) == torch.nn.Conv2d: # Conv2d is replaced
            Log.Print("Detected %s : %s"%(name+"."+attr_str, ta), current=False, elapsed=False)
            if bfpc == None:
                Log.Print("  == Didn't replaced", current=False, elapsed=False)
            else:
                setattr(net, attr_str, ReturnBFPConv2d(ta, bfpc))
                Log.Print("  => Replaced to BFPConv2d:%s"%(str(bfpc)), current=False, elapsed=False)
        # else:
        #     Log.Print("Conv2d %s"%(name+"."+attr_str), current=False, elapsed=False)

    # print(name)
    for i, n in enumerate(net.children()):
        # print(name, str(n))
        bfpc = GetValueFromBFPConf(bfp_dict, name+"."+str(i))
        if type(n) == torch.nn.Conv2d:
            Log.Print("Detected %s : %s"%(name+"."+str(i), str(n)), current=False, elapsed=False)
            if bfpc == None:
                Log.Print("  == Didn't replaced", current=False, elapsed=False)
            else:
                net[i] = ReturnBFPConv2d(net[i], bfpc)
                Log.Print("  => Replaced to BFPConv2d:%s"%(str(bfpc)), current=False, elapsed=False)
            

    # Recursive call to replace other layers
    for n, ch in net.named_children():
        ReplaceLayers(ch, bfp_dict, name+"."+n)
    if type(net) in [list, tuple, torch.nn.Sequential]:
        for i, n in enumerate(net.children()):
            ReplaceLayers(net[i], bfp_dict, name+"."+str(i))


# Load models
from model.AlexNet import AlexNetCifar
from model.ResNet import ResNet18Cifar
from model.DenseNet import DenseNet121Cifar
from model.MobileNetv1 import MobileNetv1Cifar
from model.VGG import VGG16Cifar

from model.MLPMixer import mlp_mixer_b16

import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def GetNetwork(dataset, model, num_classes = 10, bfp_conf = None, pretrained = False):
    if dataset.lower() == "imagenet":
        if model.lower() in model_names:
            if pretrained:
                net = models.__dict__[args.arch](pretrained=True)
                Log.Print("Using pretrained pytorch {model} imagenet model...", current=False, elapsed=False)
            else:
                net = models.__dict__[args.arch]()
                Log.Print("Using pytorch {model} imagenet model...", current=False, elapsed=False)
        else:
            NotImplementedError("Imagenet model {model} not defined on pytorch")
    elif dataset.lower() in ["cifar10", "cifar100"]:
        if model.lower() == "alexnet":
            net = AlexNetCifar(num_classes)
        elif model.lower() == "resnet18":
            net = ResNet18Cifar(num_classes)
        elif model.lower() == "densenet121":
            net = DenseNet121Cifar(num_classes)
        elif model.lower() == "mobilenetv1":
            net = MobileNetv1Cifar(num_classes)
        elif model.lower() == "vgg16":
            net = VGG16Cifar(num_classes)
        else:
            NotImplementedError("Model {model} on cifar10/cifar100 not implemented on ./models/ folder.")
    else:
        NotImplementedError("Dataset {datset} not supported.")

    if bfp_conf != None:
        ReplaceLayers(net, bfp_conf)
        Log.Print("Replacing model's layers to provided bfp config...", current=False, elapsed=False)
    return net


import torch.optim as optim

def GetOptimizerScheduler(net, config=None):

    Log.Print("Loading Optimizer and Scheduler",elapsed=False, current=False)

    if config == None:
        Log.Print("  Config is None, returning default optimizer and scheduler",elapsed=False, current=False)
        o = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=200)
        return o, s
    
    Log.Print("  Config is NOT None, setting custom optimizer and scheduler",elapsed=False, current=False)
    lr = config["lr-initial"] if "lr-initial" in config else 0.1
    momentum = config["momentum"] if "momentum" in config else 0.9
    weight_decay = config["weight-decay"] if "weight-decay" in config else 5e-4

    if "optimizer" in config:
        if config["optimizer"] == "SGD":
            o = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            Log.Print("  Optimizer: SGD, lr=%f, momentum=%f, weight_decay=%f"%(lr, momentum, weight_decay),elapsed=False, current=False)
        else:
            o = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            Log.Print("  Optimizer can't recognized, setting to basic optimizer",elapsed=False, current=False)
            Log.Print("  Optimizer: SGD, lr=%f, momentum=%f, weight_decay=%f"%(lr, momentum, weight_decay),elapsed=False, current=False)
    else:
        o = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # Log.Print("  Optimizer is not set, setting to basic optimizer",elapsed=False, current=False)
        Log.Print("  Optimizer: SGD, lr=%f, momentum=%f, weight_decay=%f"%(lr, momentum, weight_decay),elapsed=False, current=False)

    T_max = config["t-max"] if "t-max" in config else 200

    if "scheduler" in config:
        if config["scheduler"] == "CosineAnnealingLR":
            s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=T_max)
            Log.Print("  LR Scheduler: CosineAnnealingLR, T_max=%d"%(T_max),elapsed=False, current=False)
        else:
            Log.Print("  Scheduler can't recognized, setting to basic scheduler",elapsed=False, current=False)
            s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=T_max)
            Log.Print("  LR Scheduler: CosineAnnealingLR, T_max=%d"%(T_max),elapsed=False, current=False)
    else:
        # Log.Print("  Scheduler is not set, setting to basic scheduler",elapsed=False, current=False)
        s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=T_max)
        Log.Print("  LR Scheduler: CosineAnnealingLR, T_max=%d"%(T_max),elapsed=False, current=False)

    if "scheduler-step" in config:
        Log.Print("  %d scheduler step added on optimizer"%(config["scheduler-step"]),elapsed=False, current=False)
        for i in range(config["scheduler-step"]):
            o.step()
            s.step()

    return o, s