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


# Flatten all childrens to 1-d array
def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

# Better version of GetAttr, it supports list / sequential too
def getattr_(obj, name):
    name = name.split(".")
    for i in name:
        if i.isdigit():
            obj = obj[int(i)]
        else:
            obj = getattr(obj, i)
    return obj

# Better version of setattr, it supports list / sequential too
def setattr_(obj, name, target):
    name = name.split(".")
    if len(name) > 1:
        for i in name[:-1]:
            if i.isdigit():
                obj = obj[int(i)]
            else:
                obj = getattr(obj, i)
    setattr(obj, name[-1], target)

