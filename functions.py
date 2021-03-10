
import numpy as np
import torch

import jax.numpy as jnp

import torchvision
import torchvision.transforms as transforms

fp32_mask = [0,
    0x00400000, 0x00600000, 0x00700000, 0x00780000,
    0x007c0000, 0x007e0000, 0x007f0000, 0x007f8000,
    0x007fc000, 0x007fe000, 0x007ff000, 0x007ff800,
    0x007ffc00, 0x007ffe00, 0x007fff00, 0x007fff80,
    0x007fffc0, 0x007fffe0, 0x007ffff0, 0x007ffff8, 0x007fffff]

fp64_mask = [0,
    0x0040000000000000, 0x0060000000000000, 0x0070000000000000, 0x0078000000000000,
    0x007c000000000000, 0x007e000000000000, 0x007f000000000000, 0x007f800000000000,
    0x007fc00000000000, 0x007fe00000000000, 0x007ff00000000000, 0x007ff80000000000,
    0x007ffc0000000000, 0x007ffe0000000000, 0x007fff0000000000, 0x007fff8000000000]


# set_mantissa_tensor : set to tensor or numpy array to speicific mantissa bits 
# TODO : Set direction of grouping
def set_mantissa_tensor(inp, group_mantissa):
    inp_n = inp.numpy() # inp_n = inp # For debug,
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32) 
    # Generate mask
    r_mask = np.asarray(np.full(v.shape, 0x007fffff, dtype=np.uint32))
    # Shift to make reversed mask
    r_mask = np.right_shift(r_mask, group_mantissa)
    # Get the reversed mask
    r_mask = np.invert(r_mask)
    # And operation to remove mantissa
    r_ = np.bitwise_and(v, r_mask)
    # revert to original np.float32 
    r = np.frombuffer(r_, dtype=np.float32)
    return torch.from_numpy(r.reshape(inp_n.shape))

# _make_group_tensor : Group values as same exponent bits, which shifts mantissa
# TODO : Set direction of grouping
def _make_groups_tensor(inp, group_mantissa, group_size, group_direction):
    inp_n = inp.numpy() # inp_n = inp # For debug,
    # Transpose to replace direction
    inp_n = np.transpose(inp_n, group_direction) # (2,3,0,1)=kernel_input_output
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32) 
    # Extract exponent
    e_mask = np.full(v.shape, 0x7f800000, dtype=np.uint32)
    e_ = np.bitwise_and(v, e_mask)
    # Get the max value
    # IDEA : send shift code to back, maybe that's faster
    np.right_shift(e_, 23, out=e_)
    # Match shape to divisible to group size
    m_ = np.append(e_, np.zeros(group_size - e_.shape[0] % group_size, dtype=np.uint32))
    m_ = np.reshape(m_, (group_size, -1))
    # get the max value of each blocks
    m_ = np.amax(m_, axis=0)
    # Revert back to original size
    m_ = np.repeat(m_, group_size)
    # Match shape back to input
    m_ = m_[:e_.shape[0]]
    # Difference of the exponent
    e_ = group_mantissa - (m_ - e_)
    # Clip the negative value (I know this is not smarter way)
    e_[e_ > 0xff] = 0
    # np.clip(e_, 0, 0xff, out=e_) # Options...
    r_mask = np.full(v.shape, 0x007fffff, dtype=np.uint32)
    # Shift to make reversed mask
    np.right_shift(r_mask, e_, out=r_mask)
    # Get the reversed mask
    np.invert(r_mask, out=r_mask)
    r_ = np.bitwise_and(v, r_mask)
    # revert to original np.float32 
    r = np.frombuffer(r_, dtype=np.float32)
    return torch.from_numpy(np.transpose(r.reshape(inp_n.shape),group_direction))

# make_group_tensor : Group values as same exponent bits, which shifts mantissa
# TODO : Set direction of grouping
def make_groups_tensor(inp, group_mantissa, group_size, group_direction):
    inp_n = inp.numpy() # inp_n = inp # For debug,
    # Transpose to replace direction
    inp_n = np.transpose(inp_n, group_direction) # (2,3,0,1)=kernel_input_output
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32)
    # Convert to jnp
    v = jnp.asarray(v)
    # Extract exponent
    e_mask = jnp.asarray(np.full(v.shape, 0x7f800000, dtype=np.int32))
    e_ = jnp.bitwise_and(v, e_mask)
    # Get the max value
    # IDEA : send shift code to back, maybe that's faster
    e_ = jnp.right_shift(e_, 23)
    # Match shape to divisible to group size
    m_ = jnp.append(e_, jnp.zeros(group_size - e_.shape[0] % group_size, dtype=np.int32))
    m_ = jnp.reshape(m_, (group_size, -1))
    # get the max value of each blocks
    m_ = jnp.amax(m_, axis=0)
    # Revert back to original size
    m_ = jnp.repeat(m_, group_size)
    # Match shape back to input
    m_ = m_[:e_.shape[0]]
    # Difference of the exponent
    # IEEE's basic mantissa bit has to be included to the value, so...
    e_ = group_mantissa - (m_ - e_)
    # Clip the negative value (I know this is not smarter way)
    e_ = jnp.clip(e_, 0, 0xfff) # np method : e_[e_ > 0xff] = 0
    # np.clip(e_, 0, 0xff, out=e_) # Options...
    r_mask = jnp.asarray(np.full(v.shape, 0x007fffff, dtype=np.uint32))
    # Shift to make reversed mask
    r_mask = jnp.right_shift(r_mask, e_)
    # Get the reversed mask
    r_mask = jnp.invert(r_mask)
    r_ = jnp.bitwise_and(v, r_mask)
    r_ = np.asarray(r_)
    # revert to original np.float32 
    r = np.frombuffer(r_, dtype=np.float32)
    r = np.transpose(r.reshape(inp_n.shape),group_direction)
    return torch.from_numpy(r)

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

def Str2Tuple(v):
    r = []
    v = v.replace(" ","").replace("(","").replace(")","").split(",")
    for i in v:
        r.append(int(i))
    return tuple(r)

class BFConf():

    def __init__(self, dic):
        self.f_i_bit = dic["f_i_bit"]            if "f_i_bit" in dic.keys() else 8
        self.f_i_sz  = dic["f_i_sz"]             if "f_i_sz"  in dic.keys() else 36
        self.f_i_dir = Str2Tuple(dic["f_i_dir"]) if "f_i_dir" in dic.keys() else (2,3,0,1)
        self.f_w_bit = dic["f_w_bit"]            if "f_w_bit" in dic.keys() else self.f_i_bit
        self.f_w_sz  = dic["f_w_sz"]             if "f_w_sz"  in dic.keys() else self.f_i_sz
        self.f_w_dir = Str2Tuple(dic["f_w_dir"]) if "f_w_dir" in dic.keys() else self.f_i_dir
        self.b_o_bit = dic["b_o_bit"]            if "b_o_bit" in dic.keys() else self.f_i_bit
        self.b_o_sz  = dic["b_o_sz"]             if "b_o_sz"  in dic.keys() else self.f_i_sz
        self.b_o_dir = Str2Tuple(dic["b_o_dir"]) if "b_o_dir" in dic.keys() else self.f_i_dir
        
    def __repr__(self):
        return str(self)
    def __str__(self):
        if self.f_i_bit == self.f_w_bit and self.f_w_bit == self.b_o_bit \
            and self.f_i_sz == self.f_w_sz and self.f_w_sz == self.b_o_sz \
            and self.f_i_dir == self.f_w_dir and self.f_w_dir == self.b_o_dir:
            return 'bit={}, size={}, dir={}'.format(self.f_i_bit, self.f_i_sz, self.f_i_dir)
        else:
            return 'bit={}/{}/{}, size={}/{}/{}, dir={}/{}/{}'.format(self.f_i_bit, self.f_w_bit, self.b_o_bit, self.f_i_sz, self.f_w_sz, self.b_o_sz, self.f_i_dir, self.f_w_dir, self.b_o_dir)
    

# Argument parsing
import argparse

def str2bool(v):
    if v.lower() in ["true", "t", "1"]: return True
    elif v.lower() in ["false", "f", "0"]: return False
    else: raise argparse.ArgumentTypeError("Not Boolean value")


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


from net import SimpleNet, ResNet18, BFSimpleNet, BFResNet18
import torch.optim as optim
import torch.nn as nn
import os
import json

# Parse arguments
def ArgumentParse(logfileStr):
    parser = argparse.ArgumentParser()
    # Data loader
    parser.add_argument("-d","--dataset", type=str, default = "CIFAR-10",
        help = "Dataset to use [CIFAR-10, CIFAR-100]")
    parser.add_argument("-nw","--num-workers", type=int, default = 8,
        help = "Number of workers to load data")

    # Model setup
    parser.add_argument("-m","--model", type=str, default = "Resnet18",
        help = "Model to use [SimpleNet, Resnet18]")
        
    parser.add_argument("-bf", "--bf-layer-conf-file", type=str, default="",
        help = "Config of the bf setup, if not set, original network will be trained")
    parser.add_argument("--cuda", type=str2bool, default=True,
        help = "Using CUDA to compute on GPU [True False]")
    
    # Training setup
    parser.add_argument("--training-epochs", type=int, default = 0,
        help = "[OVERRIDE] If larger than 0, epoch is set to this value")
    parser.add_argument("--initial-lr", type=float, default = 0,
        help = "[OVERRIDE] If larger than 0, initial lr is set to this value")
    parser.add_argument("--momentum", type=float, default = 0,
        help = "[OVERRIDE] If larger than 0, momentum is set to this value")


    # Block setup

    # Printing / Logger / Stat
    parser.add_argument("--print-train-batch", type=int, default = 0,
        help = "Print info on # of batches, 0 to disable") # 128 = 391
    parser.add_argument("--print-train-count", type=int, default = 5,
        help = "How many print on each epoch, 0 to disable") # 128 = 391
    parser.add_argument("--stat", type=str2bool, default = False,
        help = "Record to stat object?")
    parser.add_argument("--stat-loss-batches", type=int, default = 0,
        help = "[OVERRIDE] Average batches to calculate running loss on stat object")

    parser.add_argument("--save", type=str2bool, default = False,
        help = "Save best model's weight")
    
    # Parse arguments
    args = parser.parse_args()

    # Save log file location
    args.log_file_location = logfileStr
    
    # Load train data
    args.trainset, args.testset, args.classes = LoadDataset(args.dataset)

    # Parse bf layer conf from file
    if args.bf_layer_conf_file == "":
        print("bf layer confing file not set, original network will be trained.")
        args.bf_layer_conf = None
    elif not os.path.exists("./conf/"+args.bf_layer_conf_file+".json"):
        raise FileNotFoundError(args.bf_layer_conf_file + ".json not exists on ./conf/ directory!")
    else:
        f = open("./conf/"+args.bf_layer_conf_file+".json","r",encoding="utf-8")

        args.bf_layer_conf = json.load(f)
        if args.bf_layer_conf["name"] != args.model:
            raise ValueError("BF layer conf is not match with model")
    
    # Define the network and optimize almost everything
    # Simplenet, 3 convs and 3 fc layers
    if args.model == "SimpleNet":
        # Network construction
        if args.bf_layer_conf is not None:
            args.net = BFSimpleNet(args.bf_layer_conf, len(args.classes))
        else:
            args.net = SimpleNet(len(args.classes))

        # Trainloader and Testloader
        args.batch_size_train = 4
        args.batch_size_test = 100
        
        # Critertion, optimizer, scheduler
        args.criterion = nn.CrossEntropyLoss()
        args.optimizer = optim.SGD(args.net.parameters(), lr=0.001, momentum=0.9)
        args.scheduler = None

        # Training Epochs
        if args.training_epochs == 0:
            args.training_epochs = 5
        
        # Logger, stat, etc
        args.stat_loss_batches = 1000
        # args.print_train_batch = 2500

    # Resnet18, 3 convs and 3 fc layers
    elif args.model == "Resnet18":
        # Network construction
        if args.bf_layer_conf is not None:
            args.net = BFResNet18(args.bf_layer_conf, len(args.classes))
        else:
            args.net = ResNet18(len(args.classes))

        # Trainloader and Testloader
        args.batch_size_train = 128
        args.batch_size_test = 100
        
        # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        # Critertion, optimizer, scheduler
        args.criterion = nn.CrossEntropyLoss()
        args.optimizer = optim.SGD(args.net.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=5e-4)
        args.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.optimizer, T_max=200)
        
        # Training Epochs
        if args.training_epochs == 0:
            args.training_epochs = 200

        # Logger, stat, etc
        args.stat_loss_batches = 100
    else:
        raise NotImplementedError("Model {} not Implemented".format(args.model))


    # Testloader and Trainloader
    args.trainloader = torch.utils.data.DataLoader(args.trainset,
        batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    args.testloader = torch.utils.data.DataLoader(args.testset,
        batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)

    # Move model to gpu if gpu is available
    if args.cuda:
        args.net.to('cuda')
        # Temporally disabled because of JAX
        # args.net = torch.nn.DataParallel(args.net) 

    # Count of the mini-batches
    args.batch_count = len(args.trainloader)

    # Stat object
    if args.stat:
        args.stat = Stat(args)
    else:
        args.stat = None

    return args