from bfp.module import BFPConv2d, BFPLinear
from bfp.internal import get_zse
from bfp.conf import BFPConf
import torch
import numpy as np
from utils.logger import Log

def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res


def GetBFLayers(net, name="net"):
    # print(type(net), name)
    res = []
    for n, ch in net.named_children():
        # if type(ch) in [torch.nn.Conv2d, torch.nn.Linear, BFPConv2d, BFPLinear]:
        if type(ch) in [BFPConv2d, BFPLinear]:
            # Log.Print("Detected(N) %s : %s"%(name+"."+n, ch), current=False, elapsed=False)
            res.append(name+"."+n)
        l = GetBFLayers(ch, name + "." + n)
        for i in l:
            res.append(i)
    return res

FB16_CONF = '''{
        "fw_bit":8,
        "bwo_bit":8,
        "bwi_bit":8,
        "fw_dim":[1,24,3,3],
        "fi_dim":[1,6,3,3],
        "bwo_dim":[1,1,18,3],
        "bwi_dim":[1,1,18,3]
    }
'''
FB12_CONF = '''{
        "fw_bit":4,
        "bwo_bit":4,
        "bwi_bit":4,
        "fw_dim":[1,24,3,3],
        "fi_dim":[1,6,3,3],
        "bwo_dim":[1,1,18,3],
        "bwi_dim":[1,1,18,3]
    }
'''

class LayerElement:
    def __init__(self, name, mode):
        self.name = name
        self.value = 0
        self.count = 0
        self.countTotal = 0
        self.valueTotal = 0

        self.mode = mode
    
    def Add(self, val):
        self.value += val
        self.valueTotal += val
        self.count += 1
        self.countTotal += 1

    def Segment(self):
        v = self.value
        c = self.count
        self.value = 0
        self.count = 0
        return v, v/c
    
    def Total(self):
        return self.valueTotal, self.countTotal

    def GetConf(self):
        if mode == "FB16":
            return '"' + name + '":' + FB16_CONF
        elif mode == "FB12":
            return '"' + name + '":' + FB12_CONF


def getattrBetter(obj, name):
    name = name.split(".")
    for i in name:
        if i.isdigit():
            obj = obj[int(i)]
        else:
            obj = getattr(obj, i)
    return obj


def setattrBetter(obj, name, target):
    name = name.split(".")
    if len(name) > 1:
        for i in name[:-1]:
            if i.isdigit():
                obj = obj[int(i)]
            else:
                obj = getattr(obj, i)
    setattr(obj, name[-1], target)

class DynamicOptimizer:
    def __init__(self):
        self._gradict = dict()
        self.step = 0
        self.layers = dict()
        pass

    def PreloadDict(self, net):
        # Find all layers to be replaced (BFConv2d), and add to a list
        bfl = GetBFLayers(net)
        Log.Print("Detected BFP Layers to Optimize:")
        Log.Print(str(bfl),elapsed=False, current=False)
        for i in bfl:
            self.layers[i] = LayerElement(i, "FB16")

    def AppendGrad(self, net):
        
        for key, value in self.layers.items():
            layer = getattrBetter(net, key[4:])
            # print(key, layer, layer.bfp_conf)
            value.Add(get_zse(layer.weight.grad, layer.bfp_conf.bwg_bit, layer.bfp_conf.bwg_dim))


    def FlatModel(self, net):
        # Log.Print(str(flatten(net)))
        for inst in flatten(net):
            if type(inst) == BFPConv2d:
                Log.Print(str(inst))
    
    def GetGradSegment(self, print_info=True):
        lst = []
        str = "ZSE "
        for key, value in self.layers.items():
            v, a = value.Segment()
            str += "%2.4f "%a
            lst.append(a)
        if print_info:
            Log.Print(str, elapsed=False, current=False)
        return lst

    def GetLayerNames(self):
        lst = []
        for key, value in self.layers.items():
            lst.append(key)
        return lst


    def _GradAvg(self):
        st = "G:"
        for key, value in self._gradict.items():
            st += "%2.4f "%value
        Log.Print(st, elapsed = False, current = False)
        self._gradict = dict()

    def AddGradients(self, net):
        self.step += 1
        for cnt, inst in enumerate(flatten(net)):
            if type(inst) in [BFPConv2d, torch.nn.Conv2d]:
                if str(cnt) not in self._gradict:
                    self._gradict[str(cnt)] = 0
                # Log.Print(str(inst))
                self._gradict[str(cnt)] += np.average(np.absolute(inst.weight.grad.detach().cpu()))
        """
        for name, param in net.named_parameters():
            if param.requires_grad:
                Log.Print(str(name))
                # Log.Print(str(param.data.shape))
                Log.Print(str(param.grad.shape))
                Log.Print(str(param.grad))
                Log.Print("")
        """


def Gradients(args):
    # print(args.net)
#         Log.Print(str(parameter.grad.shape))
    for name, param in args.net.named_parameters():
        if param.requires_grad:
            Log.Print(str(name))
            Log.Print(str(type(param)))
            # Log.Print(str(param.data.shape))
            Log.Print(str(param.grad.shape))
            Log.Print(str(param.grad))
            Log.Print("")



DO = DynamicOptimizer()