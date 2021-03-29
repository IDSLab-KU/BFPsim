import torch
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np

# LoadDataset : Load dataset, cifar-10 or cifar-100
def LoadDataset(name):
    if name == "CIFAR-10":
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

    elif name == "CIFAR-100":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
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
        raise NotImplementedError("Dataset {} not Implemented".format(args.dataset))
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
    def __init__(self, dic):
        # Foward - Weight
        self.f_w     = dic["f_w"]                if "f_w"     in dic.keys() else True
        self.f_w_bit = dic["f_w_bit"]            if "f_w_bit" in dic.keys() else 8
        self.f_w_sz  = dic["f_w_sz"]             if "f_w_sz"  in dic.keys() else 36
        self.f_w_dir = DIR_DICT[dic["f_w_dir"]]  if "f_w_dir" in dic.keys() else DIR_DICT["WI"]

        # Forward - Input
        self.f_i     = dic["f_i"]                if "f_i"     in dic.keys() else True
        self.f_i_bit = dic["f_i_bit"]            if "f_i_bit" in dic.keys() else self.f_w_bit
        self.f_i_sz  = dic["f_i_sz"]             if "f_i_sz"  in dic.keys() else self.f_w_sz
        self.f_i_dir = DIR_DICT[dic["f_i_dir"]]  if "f_i_dir" in dic.keys() else DIR_DICT["FC"]

        # Forward - Output
        self.f_o     = dic["f_o"]                if "f_o"     in dic.keys() else True
        self.f_o_bit = dic["f_o_bit"]            if "f_o_bit" in dic.keys() else self.f_w_bit
        self.f_o_sz  = dic["f_o_sz"]             if "f_o_sz"  in dic.keys() else self.f_w_sz
        self.f_o_dir = DIR_DICT[dic["f_o_dir"]]  if "f_o_dir" in dic.keys() else DIR_DICT["FC"]

        # Backward - Weight
        self.b_w     = dic["b_w"]                if "b_w"     in dic.keys() else True
        self.b_w_bit = dic["b_w_bit"]            if "b_w_bit" in dic.keys() else self.f_w_bit
        self.b_w_sz  = dic["b_w_sz"]             if "b_w_sz"  in dic.keys() else self.f_w_sz
        self.b_w_dir = DIR_DICT[dic["b_w_dir"]]  if "b_w_dir" in dic.keys() else DIR_DICT["WO"]

        # Backward - Input
        self.b_i     = dic["b_i"]                if "b_i"     in dic.keys() else True
        self.b_i_bit = dic["b_i_bit"]            if "b_i_bit" in dic.keys() else self.f_i_bit
        self.b_i_sz  = dic["b_i_sz"]             if "b_i_sz"  in dic.keys() else self.f_i_sz
        self.b_i_dir = DIR_DICT[dic["b_i_dir"]]  if "b_i_dir" in dic.keys() else DIR_DICT["FX"]

        # Backward - Output
        self.b_o     = dic["b_o"]                if "b_o"     in dic.keys() else True
        self.b_o_bit = dic["b_o_bit"]            if "b_o_bit" in dic.keys() else self.f_o_bit
        self.b_o_sz  = dic["b_o_sz"]             if "b_o_sz"  in dic.keys() else self.f_o_sz
        self.b_o_dir = DIR_DICT[dic["b_o_dir"]]  if "b_o_dir" in dic.keys() else DIR_DICT["FX"]

        
    def __repr__(self):
        return str(self)
    def __str__(self):
        s = "["
        s += "FW/" if self.f_w else "  /" 
        s += "FI/" if self.f_i else "  /" 
        s += "FO/" if self.f_o else "  /" 
        s += "BW/" if self.b_w else "  /" 
        s += "BI/" if self.b_i else "  /" 
        s += "BW]" if self.b_w else "  ]"
        s += ",bit="
        if self.f_w_bit == self.f_i_bit == self.f_o_bit == self.b_w_bit == self.b_i_bit == self.b_o_bit:
            s += '{},'.format(self.f_w_bit)
        else:
            s += "{}/".format(self.f_w_bit) if self.f_w else "_/" 
            s += "{}/".format(self.f_i_bit) if self.f_i else "_/" 
            s += "{}/".format(self.f_o_bit) if self.f_o else "_/" 
            s += "{}/".format(self.b_w_bit) if self.b_w else "_/" 
            s += "{}/".format(self.b_i_bit) if self.b_i else "_/" 
            s += "{}".format(self.b_o_bit)  if self.b_w else "_]"
        s += ",sz="
        if self.f_w_sz == self.f_i_sz == self.f_o_sz == self.b_w_sz == self.b_i_sz == self.b_o_sz:
            s += '{},'.format(self.f_w_sz)
        else:
            s += "{}/".format(self.f_w_sz) if self.f_w else "_/" 
            s += "{}/".format(self.f_i_sz) if self.f_i else "_/" 
            s += "{}/".format(self.f_o_sz) if self.f_o else "_/" 
            s += "{}/".format(self.b_w_sz) if self.b_w else "_/" 
            s += "{}/".format(self.b_i_sz) if self.b_i else "_/" 
            s += "{}".format(self.b_o_sz)  if self.b_w else "_]"
        s += ",dir=["
        if self.f_w_dir == self.f_i_dir == self.f_o_dir == self.b_w_dir == self.b_i_dir == self.b_o_dir:
            s += '{}'.format(DirKey(self.f_w_dir))
        else:
            s += "{}/".format(DirKey(self.f_w_dir)) if self.f_w else "_/" 
            s += "{}/".format(DirKey(self.f_i_dir)) if self.f_i else "_/" 
            s += "{}/".format(DirKey(self.f_o_dir)) if self.f_o else "_/" 
            s += "{}/".format(DirKey(self.b_w_dir)) if self.b_w else "_/" 
            s += "{}/".format(DirKey(self.b_i_dir)) if self.b_i else "_/" 
            s += "{}]".format(DirKey(self.b_o_dir))  if self.b_w else "_]"
        return s

class Stat():
    def __init__(self, args):
        self.loss = []
        self.accuracy = []
        self.running_loss = 0.0
        self.loss_count = 0
        self.loss_batches = args.stat_loss_batches
        self.file_location = args.log_file_location[:-4] + ".stat"

    def AddLoss(self, v):
        self.running_loss += v
        self.loss_count += 1
        if self.loss_count == self.loss_batches:
            self.loss.append(self.running_loss / self.loss_batches)
            self.loss_count = 0
            self.running_loss = 0.0
    
    def AddAccuracy(self, v):
        self.accuracy.append(v)

    def SaveToFile(self):
        if self.loss_count != 0:
            self.loss.append(self.running_loss / self.loss_batches)
        
        f = open(self.file_location, mode="w", newline='', encoding='utf-8')
        f.write(">Average Loss per {} batches\n".format(self.loss_batches))
        for i in self.loss:
            f.write(str(i)+"\t")
        f.write("\n")
        f.write(">Accuracy\n")
        for i in self.accuracy:
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
