from bfp.internal import get_zse
from bfp.functions import GetBFLayerNames

from utils.logger import Log
from utils.functions import getattr_

# Directly attach to the BFP Layers (Works for any other layers, too btw) 
def PrepareSegment(target, name):
    target.opt_name = name
    target.opt_v = dict()
    target.opt_l = dict()
    for n in ["fi", "fw", "fo", "bio", "biw", "big", "bwo", "bwi", "bwg"]:
        target.opt_v[n] = 0
        target.opt_l[n] = 0
    # Save values
    """
    target.opt_v_fi,  target.opt_v_fi  = 0, 0
    target.opt_v_fw,  target.opt_v_fw  = 0, 0
    target.opt_v_fo,  target.opt_v_fo  = 0, 0
    target.opt_v_bio, target.opt_v_bio = 0, 0
    target.opt_v_biw, target.opt_v_biw = 0, 0
    target.opt_v_big, target.opt_v_big = 0, 0
    target.opt_v_bwo, target.opt_v_bwo = 0, 0
    target.opt_v_bwi, target.opt_v_bwi = 0, 0
    target.opt_v_bwg, target.opt_v_bwg = 0, 0
    """

    # Save counts
    target.opt_count = 0
    target.opt_count_total = 0

    # Optimizer variables


def UpdateSegment(target, value):
    for key in value:
        target.opt_v[key] += value[key]
        target.opt_l[key] = value[key]
    target.opt_count += 1

def ResetSegment(target):
    for key in target.opt_v:
        target.opt_v[key] = 0
    target.opt_count = 0

def GetSegment(target):
    res = dict()
    for key in target.opt_v:
        res[key] = target.opt_v[key] / target.opt_count
    return target.opt_l, target.opt_count, res



CANDIDATE = ["FB24", "FB16", "FB12"]
THRESHOLD_UP = [0.65, 0.55]
THRESHOLD_DOWN = [0.45, 0.35]
HOLDING = 3

from utils.logger import rCol, tCol, bCol

def CoLoRiZeX(val, prec, format = "%2.4f"):
    # magenta - red - yellow - green - cyan - blue
    tl = [tCol['b'], tCol['bb'], tCol['c'], tCol['g'], tCol['y'], tCol['r'], tCol['r'], tCol['r'], tCol['m'], tCol['m']]
    tc = tl[int(val*len(tl))]
    tx = format%val
    if prec == 16:
        bg = bCol['g']
    elif prec == 8:
        bg = bCol['y']
    else:
        bg = bCol['r']
    return tc + bg + tx[2:] + rCol

# Much color. So colorful
def CoLoRiZe(val, format = "%2.4f"):
    # magenta - red - yellow - green - cyan - blue
    tl = [tCol['b'], tCol['bb'], tCol['c'], tCol['g'], tCol['y'], tCol['r'], tCol['r'], tCol['r'], tCol['m'], tCol['m']]
    if val > 1:
        return bCol['r'] + "XX" + rCol
    else:
        v = len(tl)-1 if int(val*len(tl)) >= len(tl) else int(val*len(tl))
        tc = tl[v]
        tx = format%val
        return tc + tx[2:] + rCol

def CoLoRiZeB(val, txt = '@'):
    if val == 16:
        bg = bCol['g']
    elif val == 8:
        bg = bCol['y']
    else:
        bg = bCol['r']

    return bg + txt + rCol



class DynamicOptimizer:
    def __init__(self):
        self.step = 0
        self.layers = dict()
        self.layerNames = list()
        self.updateCount = 0
        self.updateCountTotal = 0
        self.optimizeCount = 0
        self.optimizeStep = -1

        self.log_dir = ""

        self.optimizeArg = ""
        self.optimizeMode = ""

        self.actForwardInput = {}
        self.actForwardOutput = {}
        self.actBackwardGradInput = {}
        self.actBackwardGradOutput = {}
    

    def SetActivationForward(self, name):
        def hook(model, input, output):
            self.actForwardInput[name] = input[0].detach()
            self.actForwardOutput[name] = output.detach()
        return hook
        
    def SetActivationBackward(self, name):
        def hook(model, grad_input, grad_output):
            self.actBackwardGradInput[name] = grad_input[0].detach()
            self.actBackwardGradOutput[name] = grad_output[0].detach()
        return hook

    def Initialize(self, net, step = -1, log_dir = "", option = ""):
        Log.Print("=== DYNAMIC OPTIMIZER INITIALIZING ===", elapsed=False, current=False)
        self.layerNames = GetBFLayerNames(net)
        # Print information about detected layers
        Log.Print("Detected Layer:", elapsed=False, current=False)
        for i in self.layerNames:
            layer = getattr_(net, i[4:])
            PrepareSegment(layer, i)    
            # Register hook for save intermediate result
            layer.register_forward_hook(self.SetActivationForward(i))
            layer.register_backward_hook(self.SetActivationBackward(i))
            # Print tracked information
            s = " - " + str(i) + " : " + str(type(layer))
            Log.Print(s, elapsed=False, current=False)

        
        # Set the optimize step. every # of steps, optimizer will called
        self.optimizeStep = step
        if self.optimizeStep != -1:
            Log.Print("optimizeStep is set to %d steps."%self.optimizeStep, elapsed=False, current=False)
        else:
            Log.Print("optimizeStep is not set. Trainer need to manually call Optimize()",elapsed=False, current=False)

        # Set the log directory
        if log_dir != "":
            self.log_dir = log_dir + "/dynamic.txt"
            self.data_dir = log_dir + "/data.txt"
            Log.Print("Log will be saved to %s, %s"%(self.log_dir, self.data_dir),elapsed=False, current=False)
            self.log_file = open(self.log_dir, mode="w", newline='', encoding='utf-8')
            self.data_file = open(self.data_dir, mode="w", newline='', encoding='utf-8')

            self.log_file.write("optimizeArg:" + option + "\n")
            self.optimizeArg = option
            # Write Information
            s = "\n"
            self.log_file.write(s)
            # Write Table Information
            s = "Step\t"
            for i in self.layerNames:
                for seg in ["fi", "fw", "fo", "bio", "biw", "big", "bwo", "bwi", "bwg"]:
                    s += i + "." + seg + "\t"
            s += "\n"
            self.log_file.write(s)
            self.data_file.write(s)
        Log.Print("=== DYNAMIC OPTIMIZER INITIALIZED ===", elapsed=False, current=False)

    def CoLoRpRiNt(self, net):
        # Additional printable variables
        svv = "%4d: "%self.updateCount
        for i in self.layerNames:
            layer = getattr_(net, i[4:])
            v, _, _ = GetSegment(layer)
            svv += CoLoRiZeB(layer.bfp_conf.fw_bit) + CoLoRiZe(v["fw"], "%2.2f") + CoLoRiZeB(layer.bfp_conf.biw_bit) + CoLoRiZe(v["biw"],"%2.2f") + CoLoRiZeB(layer.bfp_conf.bwo_bit) + CoLoRiZe(v["bwo"], "%2.2f") + " "
        """
        svv += "\n      "
        for i in self.layerNames:
            layer = getattr_(net, i[4:])
            _, _, v = GetSegment(layer)
            svv += CoLoRiZeB(layer.bfp_conf.fw_bit) + CoLoRiZe(v[0]) + "/" + CoLoRiZeB(layer.bfp_conf.bwg_bit) + CoLoRiZe(v[1]) + " "
        """
        print(svv)



    def Update(self, net):
        if len(self.layerNames) == 0:
            return
        self.updateCount += 1
        self.updateCountTotal += 1

        # Updates gradient and weight information of layer
        for i in self.layerNames:
            layer = getattr_(net, i[4:])
            # print(key, layer, layer.bfp_conf)
            if layer.weight.grad == None:
                Log.Print("Warning: Layer " + i + "'s gradient is NULL. Skipping...")
                continue
            v = dict()
            v["biw"] = get_zse(layer.weight.grad, layer.bfp_conf.biw_bit, layer.bfp_conf.biw_dim)
            v["fw"] = get_zse(layer.weight, layer.bfp_conf.fw_bit, layer.bfp_conf.fw_dim)
            # v["fi"] = get_zse(self.actForwardOutput[i], layer.bfp_conf.fi_bit, layer.bfp_conf.fi_dim)
            v["bwo"] = get_zse(self.actBackwardGradInput[i], layer.bfp_conf.bwo_bit, layer.bfp_conf.bwo_dim)
            # v["fo"] = get_zse(self.actForwardOutput[i], layer.bfp_conf.fo_bit, layer.bfp_conf.fo_dim)
            # v["bwi"] = get_zse(self.actBackwardGradInput[i], layer.bfp_conf.big_bit, layer.bfp_conf.big_dim)
            UpdateSegment(layer, v)

        self.CoLoRpRiNt(net)
        
        if self.optimizeStep != -1 and self.updateCount == self.optimizeStep:
            self.Optimize(net)

    def OptimizeLayer(self, layer):
        _, _, a = GetSegment(layer)

        split = self.optimizeArg.split("/")
        if self.optimizeMode == "" and split[0] == "Simple":
            self.optimizeMode = "Simple"
            self.optUpperThreshold = float(split[2])
            self.optLowerThreshold = float(split[1])
            self.optInitialBreak = int(split[3])
            self.optHoldLength = int(split[4])
        
        if self.optimizeMode == "Simple":
            if self.optimizeCount > self.optInitialBreak:
                if a["fw"] < self.optLowerThreshold:
                    if layer.bfp_conf.fw_bit == 16:
                        layer.bfp_conf.fw_bit = 8
                        layer.bfp_conf.fi_bit = 8
                    elif layer.bfp_conf.fw_bit == 8:
                        layer.bfp_conf.fw_bit = 4
                        layer.bfp_conf.fi_bit = 4
                elif a["fw"] > self.optUpperThreshold:
                    if layer.bfp_conf.fw_bit == 4:
                        layer.bfp_conf.fw_bit = 8
                        layer.bfp_conf.fi_bit = 8
                    elif layer.bfp_conf.fw_bit == 8:
                        layer.bfp_conf.fw_bit = 16
                        layer.bfp_conf.fi_bit = 16
                if a["biw"] < self.optLowerThreshold:
                    if layer.bfp_conf.biw_bit == 16:
                        layer.bfp_conf.biw_bit = 8
                        layer.bfp_conf.bio_bit = 8
                    elif layer.bfp_conf.biw_bit == 8:
                        layer.bfp_conf.biw_bit = 4
                        layer.bfp_conf.bio_bit = 4
                elif a["biw"] > self.optUpperThreshold:
                    if layer.bfp_conf.biw_bit == 4:
                        layer.bfp_conf.biw_bit = 8
                        layer.bfp_conf.bio_bit = 8
                    elif layer.bfp_conf.biw_bit == 8:
                        layer.bfp_conf.biw_bit = 16
                        layer.bfp_conf.bio_bit = 16
                if a["bwo"] < self.optLowerThreshold:
                    if layer.bfp_conf.bwo_bit == 16:
                        layer.bfp_conf.bwo_bit = 8
                        layer.bfp_conf.bwi_bit = 8
                    elif layer.bfp_conf.bwo_bit == 8:
                        layer.bfp_conf.bwo_bit = 4
                        layer.bfp_conf.bwi_bit = 4
                elif a["bwo"] > self.optUpperThreshold:
                    if layer.bfp_conf.bwo_bit == 4:
                        layer.bfp_conf.bwo_bit = 8
                        layer.bfp_conf.bwi_bit = 8
                    elif layer.bfp_conf.bwo_bit == 8:
                        layer.bfp_conf.bwo_bit = 16
                        layer.bfp_conf.bwi_bit = 16

    def Optimize(self, net):
        Log.Print("=== OPTIMIZING STEP %d ==="%self.optimizeCount, elapsed=False, current=False)
        # Optimizes one step further
        self.optimizeCount += 1
        
        # Optimize Method

        for i in self.layerNames:
            layer = getattr_(net, i[4:])
            prev_fw_bit = layer.bfp_conf.fw_bit
            prev_bwg_bit = layer.bfp_conf.bwg_bit
            _, _, a = GetSegment(layer)
            s = " - " + str(i) + " : "
            for key in a:
                s += "%2.4f/"%a[key]
            self.OptimizeLayer(layer)
            """
            if prev_fw_bit != layer.bfp_conf.fw_bit:
                s += "fw %2d->%2d"%(prev_fw_bit, layer.bfp_conf.fw_bit)
            if prev_bwg_bit != layer.bfp_conf.bwg_bit:
                s += "bwg %2d->%2d"%(prev_bwg_bit, layer.bfp_conf.bwg_bit)
            """
            Log.Print(s, elapsed=False, current=False)

        # Show overall
        sl = str(self.optimizeCount) + "\t"
        sd = str(self.optimizeCount) + "\t"
        for i in self.layerNames:
            layer = getattr_(net, i[4:])
            sl += "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t"%(layer.bfp_conf.fi_bit,layer.bfp_conf.fw_bit,layer.bfp_conf.fo_bit,layer.bfp_conf.bio_bit,layer.bfp_conf.biw_bit,layer.bfp_conf.big_bit,layer.bfp_conf.bwo_bit,layer.bfp_conf.bwi_bit,layer.bfp_conf.bwg_bit)
            _, _, a = GetSegment(layer)
            for key in a:
                sd += "%.4f\t"%a[key]
            # sd += "%.4f\t%.4f\t"%(a["fi"],a["fw"],a["fo"],a["bio"])
        sl += "\n"
        sd += "\n"
        if self.log_dir != "":
            self.log_file.write(sl)
            self.data_file.write(sd)
        Log.Print(sl.replace("\t"," "), end="", elapsed=False, current=False)

        for i in self.layerNames:
            layer = getattr_(net, i[4:])
            ResetSegment(layer)
        self.updateCount = 0


DO = DynamicOptimizer()