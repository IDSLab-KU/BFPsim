"""
    This code is part of the BFPSim (https://github.com/ids-Lab-DGIST/BFPSim)

    Seunghyun Lee (R3C0D3r) from IDSLab, DGIST
    coder@dgist.ac.kr

    License: CC BY 4.0
"""

import torch
import os
import json

from bfp.conf import BFPConf
from bfp.module import BFPLinear, BFPConv2d

# If you are migrating this code to separate library, plase remove line with "Log.Print" and next line
from utils.logger import Log

CONF_NET_PATH = "./conf_net/"

def LoadBFPDictFromFile(file):
    if file == "":
        Log.Print("bf layer config file not set, returning empty dict. This will generate unchanged model", current=False, elapsed=False)
        conf = dict()
    elif not os.path.exists(CONF_NET_PATH+file+".json"):
        raise FileNotFoundError("%s.json not exists on %s directory!"%(file, CONF_NET_PATH))
        # Log.Print(file + ".json not found, returning empty bf_conf_dict...", current=False, elapsed=False)
        # return dict()
    else:
        with open(CONF_NET_PATH+file+".json","r",encoding="utf-8") as f:
            conf = json.load(f)
    return conf

def GetValueFromDict(bfp_dict, attr_str):
    if attr_str in bfp_dict: # Layer configuration is found
        # If type is normal Conv2d
        if "type" in bfp_dict[attr_str] and bfp_dict[attr_str]["type"] in ["torch.nn.Conv2d", "torch.nn.Linear"]:
            return None  
        else:   # Found Config!
            return BFPConf(bfp_dict[attr_str])
    elif "default" in bfp_dict: # If default value is set, use the default value
        return BFPConf(bfp_dict["default"])
    else: # If no default value is set, don't replace
        return None

def ReturnBFPConv2d(ta, bfpc):
    if bfpc == None:
        return None
    bias = True if ta.bias != None else False
    new = BFPConv2d(in_channels=ta.in_channels, out_channels=ta.out_channels, kernel_size=ta.kernel_size, bfp_conf=bfpc, stride=ta.stride, padding=ta.padding, dilation=ta.dilation, groups=ta.groups, bias=bias, padding_mode=ta.padding_mode)
    return new

def ReturnBFPLinear(ta, bfpc):
    if bfpc == None:
        return None
    bias = True if ta.bias != None else False
    new = BFPLinear(in_features = ta.in_features, out_features=ta.out_features, bfp_conf=bfpc, bias=bias)
    return new

def GetValueFromDict(bfp_dict, attr_str):
    
    if attr_str in bfp_dict: # Layer configuration is found
        if "type" in bfp_dict[attr_str] and bfp_dict[attr_str]["type"] in ["torch.nn.Conv2d", "torch.nn.Linear", "default"]:
            return None
        else:   # Found Config!
            return BFPConf(bfp_dict[attr_str])
    elif "default" in bfp_dict: # If default value is set, use the default value
        return BFPConf(bfp_dict["default"])
    else: # If no default value is set, don't replace
        return None

def ReplaceLayers(net, bfp_dict, name="net"):
    # Log.Print("%s / %s"%(type(net),name), current=False, elapsed=False)
    for attr_str in dir(net):
        # Get the Attributes
        attr_value = getattr(net, attr_str)
        if attr_str == "zero_grad":
            continue
        if type(attr_value) in [torch.nn.Conv2d, torch.nn.Linear]:
            Log.Print("Detected(N) %s : %s"%(name+"."+attr_str, attr_value), current=False, elapsed=False)
            bfpc = GetValueFromDict(bfp_dict, name+"."+attr_str)
            if bfpc != None:
                # Replace Actual
                if type(attr_value) == torch.nn.Conv2d:
                    setattr(net, attr_str, ReturnBFPConv2d(attr_value, bfpc))
                elif type(attr_value) == torch.nn.Linear:
                    setattr(net, attr_str, ReturnBFPLinear(attr_value, bfpc))
                Log.Print("  => Replaced : %s"%(str(bfpc)), current=False, elapsed=False)
            else:
                Log.Print("  == Didn't replaced", current=False, elapsed=False)
    
    # Log.Print("Child @ %s"%name, current=False, elapsed=False)
    for n, ch in net.named_children():
        ReplaceLayers(ch, bfp_dict, name+"."+n)
    # Log.Print("Iter @ %s"%name, current=False, elapsed=False)
    if type(net) in [list, tuple, torch.nn.Sequential]:
        for i, n in enumerate(net.children()):
            if type(net[i]) in [torch.nn.Conv2d, torch.nn.Linear]:
                Log.Print("Detected(I) %s : %s"%(name+"."+str(i), n), current=False, elapsed=False)
                bfpc = GetValueFromDict(bfp_dict, name+"."+str(i))
                if bfpc != None:
                    # Replace Actual
                    # TODO : Check if this is works or not
                    if type(n) == torch.nn.Conv2d:
                        net[i] = ReturnBFPConv2d(n, bfpc)
                    elif type(n) == torch.nn.Linear:
                        net[i] = ReturnBFPLinear(n, bfpc)
                    Log.Print("  => Replaced : %s"%(str(bfpc)), current=False, elapsed=False)
                else:
                    Log.Print("  == Didn't replaced", current=False, elapsed=False)

            ReplaceLayers(net[i], bfp_dict, name+"."+str(i))

    # Log.Print("End @ %s"%name, current=False, elapsed=False)


"""
Example of Using ReplaceLayers()

# It is possible to make bpf_dict by own
bfp_dict = dict()
# Define default value
bfp_dict["default"] = BFPConf()
bfp_dict["net.conv1"] = BFPConf()
ReplaceLayers(net, bfp_dict)

# Or, you can provide the file's path
path = "default_FB12_WG24"
bfp_dict = LoadBFPDictFromFile(path)
ReplaceLayers(net, bfp_dict)

"""
