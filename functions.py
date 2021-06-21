import torch
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np

# LoadDataset : Load dataset, cifar-10 or cifar-100
def LoadDataset(name):
    if name == "CIFAR10":
        # Set transform
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        # Prepare Cifar-10 Dataset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_train)
        testset =  torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif name == "CIFAR100":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset =  torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        
        classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
                'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
                'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
                'bottle', 'bowl', 'can', 'cup', 'plate',
                'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
                'clock', 'keyboard', 'lamp', 'telephone', 'television',
                'bed', 'chair', 'couch', 'table', 'wardrobe',
                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                'bear', 'leopard', 'lion', 'tiger', 'wolf',
                'bridge', 'castle', 'house', 'road', 'skyscraper',
                'cloud', 'forest', 'mountain', 'plain', 'sea',
                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                'crab', 'lobster', 'snail', 'spider', 'worm',
                'baby', 'boy', 'girl', 'man', 'woman',
                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
                'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
                'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    else:
        raise NotImplementedError("Dataset {} not Implemented".format(name))
    return trainset, testset, classes

def str2tuple(v):
    r = []
    v = v.replace(" ","").replace("(","").replace(")","").split(",")
    for i in v:
        r.append(int(i))
    return tuple(r)

def str2bool(v):
    if v.lower() in ["true", "t", "1"]: return True
    elif v.lower() in ["false", "f", "0"]: return False
    else: raise argparse.ArgumentTypeError("Not Boolean value")

DIR_DICT = {
    "WI" :  0,
    "WO" :  1,
    "FX" : 10,
    "FY" : 11,
    "FC" : 12
}

def DirKey(v):
    for key in DIR_DICT:
        if DIR_DICT[key] == v:
            return key

class BFConf():
    def __init__(self, dic, bwg_boost = 1.0):
        # Foward - Weight
        self.fw     = dic["fw"]                if "fw"     in dic.keys() else True
        self.fw_bit = dic["fw_bit"]            if "fw_bit" in dic.keys() else 8
        self.fw_sz  = dic["fw_sz"]             if "fw_sz"  in dic.keys() else 36
        self.fw_dir = DIR_DICT[dic["fw_dir"]]  if "fw_dir" in dic.keys() else DIR_DICT["WI"]

        # Forward - Input
        self.fi     = dic["fi"]                if "fi"     in dic.keys() else True
        self.fi_bit = dic["fi_bit"]            if "fi_bit" in dic.keys() else self.fw_bit
        self.fi_sz  = dic["fi_sz"]             if "fi_sz"  in dic.keys() else self.fw_sz
        self.fi_dir = DIR_DICT[dic["fi_dir"]]  if "fi_dir" in dic.keys() else DIR_DICT["FC"]

        # Forward - Output
        self.fo     = dic["fo"]                if "fo"     in dic.keys() else False
        self.fo_bit = dic["fo_bit"]            if "fo_bit" in dic.keys() else self.fw_bit
        self.fo_sz  = dic["fo_sz"]             if "fo_sz"  in dic.keys() else self.fw_sz
        self.fo_dir = DIR_DICT[dic["fo_dir"]]  if "fo_dir" in dic.keys() else DIR_DICT["FC"]

        # Backward - Output gradient while calculating input gradient
        self.bio     = dic["bio"]                if "bio"     in dic.keys() else True
        self.bio_bit = dic["bio_bit"]            if "bio_bit" in dic.keys() else self.fo_bit
        self.bio_sz  = dic["bio_sz"]             if "bio_sz"  in dic.keys() else self.fo_sz
        self.bio_dir = DIR_DICT[dic["bio_dir"]]  if "bio_dir" in dic.keys() else DIR_DICT["FC"]

        # Backward - Weight while calculating input gradient
        self.biw     = dic["biw"]                if "biw"     in dic.keys() else True
        self.biw_bit = dic["biw_bit"]            if "biw_bit" in dic.keys() else self.fw_bit
        self.biw_sz  = dic["biw_sz"]             if "biw_sz"  in dic.keys() else self.fw_sz
        self.biw_dir = DIR_DICT[dic["biw_dir"]]  if "biw_dir" in dic.keys() else DIR_DICT["WO"]

        # Backward - Calculated input gradient
        self.big     = dic["big"]                if "big"     in dic.keys() else False
        self.big_bit = dic["big_bit"]            if "big_bit" in dic.keys() else self.fi_bit
        self.big_sz  = dic["big_sz"]             if "big_sz"  in dic.keys() else self.fi_sz
        self.big_dir = DIR_DICT[dic["big_dir"]]  if "big_dir" in dic.keys() else DIR_DICT["FC"]

        # Backward - Output gradient while calculating weight gradient
        self.bwo     = dic["bwo"]                if "bwo"     in dic.keys() else True
        self.bwo_bit = dic["bwo_bit"]            if "bwo_bit" in dic.keys() else self.fo_bit
        self.bwo_sz  = dic["bwo_sz"]             if "bwo_sz"  in dic.keys() else self.fo_sz
        self.bwo_dir = DIR_DICT[dic["bwo_dir"]]  if "bwo_dir" in dic.keys() else DIR_DICT["FC"]

        # Backward - Input while calculating weight gradient
        self.bwi     = dic["bwi"]                if "bwi"     in dic.keys() else True
        self.bwi_bit = dic["bwi_bit"]            if "bwi_bit" in dic.keys() else self.fi_bit
        self.bwi_sz  = dic["bwi_sz"]             if "bwi_sz"  in dic.keys() else self.fi_sz
        self.bwi_dir = DIR_DICT[dic["bwi_dir"]]  if "bwi_dir" in dic.keys() else DIR_DICT["FC"]

        # Backward - Calculated weight gradient
        self.bwg     = dic["bwg"]                if "bwg"     in dic.keys() else False
        self.bwg_bit = dic["bwg_bit"]            if "bwg_bit" in dic.keys() else self.fw_bit
        self.bwg_sz  = dic["bwg_sz"]             if "bwg_sz"  in dic.keys() else self.fw_sz
        self.bwg_dir = DIR_DICT[dic["bwg_dir"]]  if "bwg_dir" in dic.keys() else DIR_DICT["WO"]

        self.bwg_boost = bwg_boost

    def __repr__(self):
        return str(self)
    def __str__(self):
        s = "["
        s += "FW/" if self.fw else "  /" 
        s += "FI/" if self.fi else "  /" 
        s += "FO/" if self.fo else "  /" 
        s += "BIO/" if self.bio else "   /" 
        s += "BIW/" if self.biw else "   /" 
        s += "BIG/" if self.big else "   /" 
        if (self.bwo_bit == self.bio_bit and self.bwo_sz == self.bio_sz and self.bwo_dir == self.bio_dir):
            s += "BWO*/" if self.bwo else "   /" 
        else:
            s += "BWO/" if self.bwo else "   /" 
        if (self.bwi_bit == self.fi_bit and self.bwi_sz == self.fi_sz and self.bwi_dir == self.fi_dir):
            s += "BWI*/" if self.bwi else "   /" 
        else:
            s += "BWI/" if self.bwi else "   /" 
        s += "BWG]" if self.bwg else "   ]"
        s += ",bit="
        if self.fw_bit == self.fi_bit == self.fo_bit == self.bio_bit == self.biw_bit == self.big_bit == self.bwo_bit == self.bwi_bit == self.bwg_bit:
            s += '{}'.format(self.fw_bit)
        else:
            s += "{}/".format(self.fw_bit) if self.fw else "_/" 
            s += "{}/".format(self.fi_bit) if self.fi else "_/" 
            s += "{}/".format(self.fo_bit) if self.fo else "_/" 
            s += "{}/".format(self.bio_bit) if self.bio else "_/" 
            s += "{}/".format(self.biw_bit) if self.biw else "_/" 
            s += "{}/".format(self.big_bit) if self.big else "_/" 
            s += "{}/".format(self.bwo_bit) if self.bwo else "_/" 
            s += "{}/".format(self.bwi_bit) if self.bwi else "_/" 
            s += "{}".format(self.bwg_bit)  if self.bwg else "_]"
        s += ",sz="
        if self.fw_sz == self.fi_sz == self.fo_sz == self.bio_sz == self.biw_sz == self.big_sz == self.bwo_sz == self.bwi_sz == self.bwg_sz:
            s += '{}'.format(self.fw_sz)
        else:
            s += "{}/".format(self.fw_sz) if self.fw else "_/" 
            s += "{}/".format(self.fi_sz) if self.fi else "_/" 
            s += "{}/".format(self.fo_sz) if self.fo else "_/" 
            s += "{}/".format(self.bio_sz) if self.bio else "_/" 
            s += "{}/".format(self.biw_sz) if self.biw else "_/" 
            s += "{}/".format(self.big_sz) if self.big else "_/" 
            s += "{}/".format(self.bwo_sz) if self.bwo else "_/" 
            s += "{}/".format(self.bwi_sz) if self.bwi else "_/" 
            s += "{}".format(self.bwg_sz)  if self.bwg else "_]"
        s += ",dir=["
        if self.fw_dir == self.fi_dir == self.fo_dir == self.bio_dir == self.biw_dir == self.big_dir == self.bwo_dir == self.bwi_dir == self.bwg_dir:
            s += '{}'.format(self.fw_dir)
        else:
            s += "{}/".format(DirKey(self.fw_dir)) if self.fw else "_/" 
            s += "{}/".format(DirKey(self.fi_dir)) if self.fi else "_/" 
            s += "{}/".format(DirKey(self.fo_dir)) if self.fo else "_/" 
            s += "{}/".format(DirKey(self.bio_dir)) if self.bio else "_/" 
            s += "{}/".format(DirKey(self.biw_dir)) if self.biw else "_/" 
            s += "{}/".format(DirKey(self.big_dir)) if self.big else "_/" 
            s += "{}/".format(DirKey(self.bwo_dir)) if self.bwo else "_/" 
            s += "{}/".format(DirKey(self.bwi_dir)) if self.bwi else "_/" 
            s += "{}".format(DirKey(self.bwg_dir))  if self.bwg else "_]"
        s += ",bwg_boost={}".format(self.bwg_boost)
        return s

class Stat():
    def __init__(self, args):
        self.loss = []
        self.testAccuracy = []
        self.trainAccuracy = []
        self.running_loss = 0.0
        self.loss_count = 0
        self.loss_batches = args.stat_loss_batches
        self.file_location = args.stat_location

    def AddLoss(self, v):
        self.running_loss += v
        self.loss_count += 1
        if self.loss_count == self.loss_batches:
            self.loss.append(self.running_loss / self.loss_batches)
            self.loss_count = 0
            self.running_loss = 0.0
    
    def AddTestAccuracy(self, v):
        self.testAccuracy.append(v)

    def AddTrainAccuracy(self, v):
        self.trainAccuracy.append(v)

    def SaveToFile(self):
        if self.loss_count != 0:
            self.loss.append(self.running_loss / self.loss_batches)
        
        f = open(self.file_location, mode="w+", newline='', encoding='utf-8')
        f.write(">Average Loss per {} batches\n".format(self.loss_batches))
        for i in self.loss:
            f.write(str(i)+"\t")
        f.write("\n")
        f.write("> Test Accuracy\n")
        for i in self.testAccuracy:
            f.write(str(i)+"\t")
        f.write("\n")
        if len(self.trainAccuracy) > 0:
            f.write("> Train Accuracy\n")
            for i in self.trainAccuracy:
                f.write(str(i)+"\t")
            f.write("\n")


from matplotlib import pyplot as plt

def SaveStackedGraph(xlabels, data, mode="percentage", title="", save=""):
    if mode == "percentage":
        percent = data / data.sum(axis=0).astype(float) * 100
    else:
        percent = data
    
    # Set figure
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    x = np.arange(data.shape[1])
    # Set the colors
    colors = ['#f00']
    for i in range(data.shape[0]):
        colors.append("{}".format(0.5 - 0.5 * float(i) / float(data.shape[0])))
    ax.stackplot(x, percent, colors=colors)
    ax.set_title(title)
    # Set labels
    if mode == "percentage":
        ax.set_ylabel('Percent (%)')
    else:
        ax.set_ylabel('Count')
    plt.xlabel("", labelpad=30)
    # plt.tight_layout(pad=6.0)

    # Set X labels
    plt.xticks(x,xlabels, rotation=45)
    fig.autofmt_xdate()
    ax.margins(0, 0) # Set margins to avoid "whitespace"
    
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/"+save + ".png")


from log import Log
from block import BFLinear, BFConv2d

def SetConv2dLayer(name, bf_conf, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode="zeros", dilation=1, groups=1, bias=True, bwg_boost=1.0):
    if name in bf_conf:
        return BFConv2d(in_channels, out_channels, kernel_size, BFConf(bf_conf[name], bwg_boost), stride, padding, dilation, groups, bias, padding_mode)
    else:
        Log.Print("WARNING(SetConv2dLayer): Name %s not in config file. Returning normal nn.Conv2d"%name, col='m', current=False, elapsed=False)
        return torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

def SaveModel(args, suffix):
    PATH = "%s_%s.model"%(args.save_prefix,suffix)
    Log.Print("Saving model file as %s"%PATH)
    torch.save(args.net.state_dict(), PATH)

from model.AlexNet import AlexNet
from model.ResNet import ResNet18
from model.DenseNet import DenseNetCifar
from model.MobileNetv1 import MobileNetv1
from model.VGG import VGG16

def GetNetwork(model, bf_layer_conf, classes):
    if model == "AlexNet":
        net = AlexNet(bf_layer_conf, len(classes))
    elif model == "ResNet18":
        net = ResNet18(bf_layer_conf, len(classes))
    elif model == "VGG16":
        net = VGG16(bf_layer_conf, len(classes))
    elif model == "MobileNetv1":
        net = MobileNetv1(bf_layer_conf, len(classes))
    elif model == "DenseNetCifar":
        net = DenseNetCifar(bf_layer_conf, len(classes))
    else:
        raise NotImplementedError("Model {} not Implemented".format(model))
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