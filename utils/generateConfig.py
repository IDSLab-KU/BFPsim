
import torch
import torch.optim as optim
import torch.nn as nn

from functions import BFConf

def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res


class ConfigLayer():
    def __init__(self):
        pass
class ConfigObj():
    def __init__(self):
        self.layers = dict()
        self.layerObjects = dict()

    def AddLayer(self, name, type, module):
        a = dict()
        a["type"] = type
        self.layers[name] = BFConf(a)
        self.layerObjects[name] = module

    def __str__(self) -> str:
        s = ""
        for key, value in self.layers.items():
            s += _Y(key) + " : %s\n  "%(self.layerObjects[key])+_W(str(value)) + "\n"
        return s

    def __repr__(self) -> str:
        return str(self)


configObj = ConfigObj()
appendString = "" # Use Global

def _Y(s):
    return '\033[33m' + s + '\033[0m'
def _R(s):
    return '\033[31m' + s + '\033[0m'
def _W(s):
    return '\033[97m' + s + '\033[0m'
def _G(s):
    return '\033[92m' + s + '\033[0m'

def ParseCommand(inp):
    inp = inp.split(" ")

    pass

def SetBit(inp):
    global appendString
def SetSize(inp):
    global appendString

def SetDir(inp):
    global appendString

def Write(inp):
    global appendString

def Load(inp):
    global appendString

def Undo(inp):
    global appendString

def GenerateConfig(args):
    global appendString
    print("------------------------------------")
    print("Named Layers")
    for name, module in args.net.named_modules():
        name = "net." + name
        if isinstance(module, nn.Linear):
            print("LINEAR: " + name)
            configObj.AddLayer(name, "Linear", module)
        if isinstance(module, nn.Conv2d):
            print("CONV2D: " + name)
            configObj.AddLayer(name, "Conv2d", module)

    # Main console entry point
    end = True
    while end:
        print("\n"*40) # Not cleaning since it can cause problem on some environments
        print("==== Current Config Object ====")
        print(configObj)
        print("Commands, see documentation to specific usage of each commands")
        print(" - "+_G("set")+_R("B")+_G("it")+" [layername=*,all] [mode=*/f/b/all] [mantissa bits]")
        print(" - "+_G("set")+_R("S")+_G("ize")+" [layername=*,all] [mode=*/f/b/all] [group size]")
        print(" - "+_G("set")+_R("D")+_G("ir")+" [layername=*,all] [mode=*/f/b/all] [WI/WO/FC/FX/FY]")
        print(" - "+_R("W")+_G("rite")+" [filename]")
        print(" - "+_R("L")+_G("oad")+" [filename]")
        print(" - "+_R("U")+_G("ndo")+" (Only once)")
        print(" - "+_R("Q")+_G("uit")+" ")

        print(appendString)
        inp = input(" >> ")
        inp = inp.split(" ")
        appendString = "\n"
        if inp[0].lower() in ["setbit", "bit", "b"]:
            if len(inp)==4:
                SetBit(inp) 
            else:
                appendString += _R("Arguments are not correct\n")
        elif inp[0].lower() in ["setsize", "size", "s"]:
            if len(inp)==4:
                SetSize(inp) 
            else:
                appendString += _R("Arguments are not correct\n")
        elif inp[0].lower() in ["setdir", "dir", "d"]:
            if len(inp)==4:
                SetDir(inp) 
            else:
                appendString += _R("Arguments are not correct\n")
        elif inp[0].lower() in ["write", "w"]:
            Write(inp)
        elif inp[0].lower() in ["load", "l"]:
            Load(inp)
        elif inp[0].lower() in ["undo", "u"]:
            Undo(inp)
        elif inp[0].lower() in ["q", "quit"]:
            end = False
        else:
            appendString += _R("Keyword unknown\n")