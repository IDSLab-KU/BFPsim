
import numpy as np
import torch

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
def make_groups_tensor(inp, group_mantissa, group_size, group_direction):
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
        # weight
        self.w_bit = dic["w_bit"]                if "w_bit" in dic.keys() else 8
        self.w_sz  = dic["w_sz"]                 if "w_sz"  in dic.keys() else 36
        self.w_dir = Str2Tuple(dic["w_dir"])     if "w_dir" in dic.keys() else (2,3,0,1)
        # forward
        self.f_i_bit = dic["f_i_bit"]            if "f_i_bit" in dic.keys() else self.w_bit
        self.f_i_sz  = dic["f_i_sz"]             if "f_i_sz"  in dic.keys() else self.w_sz
        self.f_i_dir = Str2Tuple(dic["f_i_dir"]) if "f_i_dir" in dic.keys() else self.w_dir
        self.f_o_bit = dic["f_o_bit"]            if "f_w_bit" in dic.keys() else self.f_i_bit
        # backward
        self.g_o_bit = dic["g_o_bit"]            if "g_o_bit" in dic.keys() else self.w_bit
        self.g_o_sz  = dic["g_o_sz"]             if "g_o_sz"  in dic.keys() else self.w_sz
        self.g_o_dir = Str2Tuple(dic["g_o_dir"]) if "g_o_dir" in dic.keys() else self.w_dir
        self.g_i_bit = dic["g_i_bit"]            if "g_i_bit" in dic.keys() else self.g_o_bit
        self.g_w_bit = dic["g_w_bit"]            if "g_w_bit" in dic.keys() else self.g_o_bit
        self.g_b_bit = dic["g_b_bit"]            if "g_b_bit" in dic.keys() else self.g_o_bit
        
    def __repr__(self):
        return str(self)
    def __str__(self):
        s = ""
        if self.w_bit == self.f_i_bit == self.g_o_bit \
            and self.w_sz == self.f_i_sz == self.g_o_sz \
            and self.w_dir == self.f_i_dir == self.g_o_dir:
            s += 'bit={}, size={}, dir={}'.format(self.w_bit, self.w_sz, self.w_dir)
        else:
            s += 'w_bit={}, w_sz={}, w_dir={}'.format(self.w_bit, self.w_sz, self.w_dir)
            if self.w_bit != self.f_i_bit:
                s += ', f_i_bit={}'.format(self.f_i_bit)
            if self.w_sz != self.f_i_sz:
                s += ', f_i_sz={}'.format(self.f_i_sz)
            if self.w_dir != self.f_i_dir:
                s += ', f_i_dir={}'.format(self.f_i_dir)
            if self.w_bit != self.g_o_bit:
                s += ', g_o_bit={}'.format(self.g_o_bit)
            if self.w_sz != self.g_o_sz:
                s += ', g_o_sz={}'.format(self.g_o_sz)
            if self.w_dir != self.g_o_dir:
                s += ', g_o_dir={}'.format(self.g_o_dir)
        if self.f_i_bit != self.f_o_bit:
            s += ', f_o_bit={}'.format(self.f_o_bit)
        if self.g_o_bit != self.g_i_bit:
            s += ', g_i_bit={}'.format(self.g_i_bit)
        if self.g_o_bit != self.g_w_bit:
            s += ', g_w_bit={}'.format(self.g_w_bit)
        if self.g_o_bit != self.g_b_bit:
            s += ', g_b_bit={}'.format(self.g_b_bit)
        return s

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