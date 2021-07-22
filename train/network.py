
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


def GetOptimizer(args, epoch):
    if str(epoch) in args.optimizer_dict:
        Log.Print("Setting optimizer from dict", elapsed=False, current=False)
        config = args.optimizer_dict[str(epoch)]
        lr = config["lr-initial"] if "lr-initial" in config else args.optim_lr
        momentum = config["momentum"] if "momentum" in config else args.optim_momentum
        weight_decay = config["weight-decay"] if "weight-decay" in config else args.optim_weight_decay
    else:
        Log.Print("Configuration not found. Returning default Optimizer...", elapsed=False, current=False)
        lr = args.optim_lr
        momentum = args.optim_momentum
        weight_decay = args.optim_weight_decay

    opt = optim.SGD(args.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if str(epoch) in args.optimizer_dict:
        if "step" in config:
            for i in range(config["step"]):
                opt.step()

    return opt

# Scheduler also uses same dict with optimizer
def GetScheduler(args, epoch):
    if str(epoch) in args.optimizer_dict:
        Log.Print("Setting scheduler from dict", elapsed=False, current=False)
        config = args.optimizer_dict[str(epoch)]
    else:
        Log.Print("Configuration not found. Returning default Scheduler from args...", elapsed=False, current=False)
        
    sche = optim.lr_scheduler.CosineAnnealingLR(args.optimizer, T_max=args.training_epochs)
        
    if str(epoch) in args.optimizer_dict:
        config = args.optimizer_dict[str(epoch)]
        if "step" in config:
            for i in range(config["step"]):
                sche.step()

    return sche
