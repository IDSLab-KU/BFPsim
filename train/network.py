
from utils.logger import Log

import torch

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

from bfp.functions import ReplaceLayers

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
