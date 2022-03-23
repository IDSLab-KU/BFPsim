from bfp.module import BFPConv2d, BFPLinear
from bfp.internal import get_zse
from bfp.conf import BFPConf
from bfp.functions import LoadBFPDictFromFile
from train.network import GetNetwork, GetOptimizer, GetScheduler, GetDefOptimizer, GetDefScheduler
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
        self.zseBorder = 0
    
    def UpdateStep(self, print_level = 2):
        prevmode = self.mode
        prevlev = self.zseBorder
        st = ""
        _, v = self.GetSegment()
        if v < 0.45:
            if self.mode != "FB12":
                self.zseBorder -= 1
                st = "-- "
            else:
                st = "M- "
        elif v > 0.65:
            if self.mode != "FB16":
                self.zseBorder += 1
                st = "++ "
            else:
                st = "M+ "
        else:
            st = "   "

        st += "N:" + str(self.name) + " " +  " V:%2.4f"%(v) + " "

        threshold = 4
        if self.zseBorder <= -threshold:
            if self.mode == "FB24":
                self.mode = "FB16"
            elif self.mode == "FB16":
                self.mode = "FB12"
        if self.zseBorder >= threshold:
            if self.mode == "FB12":
                self.mode = "FB16"
            # elif self.mode == "FB16":
            #     self.mode = "FB24"
        
        if prevmode != self.mode:
            st += prevmode + "->" + self.mode
            self.zseBorder = 0
        else:
            st += " L:" + str(self.zseBorder)
        
        if print_level == 0:
            pass
        elif print_level == 1: # Only print when precision change
            if prevmode != self.mode:
                Log.Print(st, elapsed = False, current = False)
        elif print_level == 2:
            if prevlev != self.zseBorder:
                Log.Print(st, elapsed = False, current = False)
        else:
            Log.Print(st, elapsed = False, current = False)


    def Add(self, val):
        self.value += val
        self.valueTotal += val
        self.count += 1
        self.countTotal += 1

    def GetSegment(self):
        v = self.value
        c = self.count
        return v, v/c
    
    def ResetSegment(self):
        self.value = 0
        self.count = 0
    
    def Total(self):
        return self.valueTotal, self.countTotal

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
        self.step = 0
        self.layers = dict()

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

    def GetGradSegment(self, print_info=True):
        lst = []
        str = "ZSE "
        for key, value in self.layers.items():
            v, a = value.GetSegment()
            str += "%2.4f "%a
            lst.append(a)
        if print_info:
            Log.Print(str, elapsed=False, current=False)
        return lst

    def ResetGradSegment(self):
        for key, value in self.layers.items():
            value.ResetSegment()


    def GetLayerNames(self):
        lst = []
        for key, value in self.layers.items():
            lst.append(key)
        return lst

    def ReplaceModel(self, args, epoch_current):
        Log.Print("==== ReplaceModel @ Epoch " + str(epoch_current) + " ====", elapsed = False, current = False)
        bfp_dict = LoadBFPDictFromFile(args.bfp_layer_conf_file)
        # Adjust model information
        initial_preserve = 1
        if epoch_current < initial_preserve:
            Log.Print("First " + str(initial_preserve) + " epoch(s): preserve precision", elapsed = False, current = False)
        else:
            for key, value in self.layers.items():
                value.UpdateStep()

        # Add bfp_dict in each stuffs
        ss = ""
        for key, value in self.layers.items():
            # Log.Print("Layer " + str(key) + " : " + str(value.mode), elapsed = False, current = False)
            bfp_dict[key] = dict()
            if value.mode == "FB24":
                bfp_dict[key]["fw_bit"] = 16
                bfp_dict[key]["bwo_bit"] = 16
                bfp_dict[key]["bwi_bit"] = 16
                bfp_dict[key]["bwg_bit"] = 16
            elif value.mode == "FB16":
                bfp_dict[key]["fw_bit"] = 8
                bfp_dict[key]["bwo_bit"] = 8
                bfp_dict[key]["bwi_bit"] = 8
                bfp_dict[key]["bwg_bit"] = 8
            elif value.mode == "FB12":
                bfp_dict[key]["fw_bit"] = 4
                bfp_dict[key]["bwo_bit"] = 4
                bfp_dict[key]["bwi_bit"] = 4
                bfp_dict[key]["bwg_bit"] = 4
                
            bfp_dict[key]["fw_dim"] = [1,24,3,3]
            bfp_dict[key]["fi_dim"] = [1,6,3,3]
            bfp_dict[key]["bwo_dim"] = [1,1,18,3]
            bfp_dict[key]["bwi_dim"] = [1,1,18,3]
            bfp_dict[key]["bwg_dim"] = [16,1,3,3]

            ss += value.mode + " "
        Log.Print(ss, elapsed=False, current=False)

        # Change Model to desired 
        net_ = args.net
        args.net = GetNetwork(args.dataset, args.model, args.num_classes, bfp_dict, silence=True)
        args.net.load_state_dict(net_.state_dict())
        args.net.eval()

        # Load Optimizer, Scheduler, stuffs
        # Fixed Step
        args.optimizer = GetDefOptimizer(args, epoch_current)
        args.scheduler = GetDefScheduler(args, epoch_current)
        if args.cuda:
            args.net.to('cuda')
        
        bfp_dict = dict()
        Log.Print("==== ReplaceModel ====", elapsed = False, current = False)


    """
    def AddGradients(self, net):
        self.step += 1
        for cnt, inst in enumerate(flatten(net)):
            if type(inst) in [BFPConv2d, torch.nn.Conv2d]:
                if str(cnt) not in self._gradict:
                    self._gradict[str(cnt)] = 0
                # Log.Print(str(inst))
                self._gradict[str(cnt)] += np.average(np.absolute(inst.weight.grad.detach().cpu()))
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