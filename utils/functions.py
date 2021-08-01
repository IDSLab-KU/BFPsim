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

def DictKey(d, v):
    for key in d:
        if d[key] == v:
            return key
