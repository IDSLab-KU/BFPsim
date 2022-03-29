from bfp.internal import get_zse
from bfp.functions import GetBFLayerNames

from utils.logger import Log
from utils.functions import getattr_

# Directly attach to the BFP Layers (Works for any other layers, too btw) 
def PrepareSegment(target, name):
    target.opt_name = name
    # Save values
    target.opt_value_weight = 0
    target.opt_value_grad = 0
    target.opt_value_weight_last = 0
    target.opt_value_grad_last = 0

    # Save counts
    target.opt_count = 0
    target.opt_count_total = 0

    # Optimizer variables


def UpdateSegment(target, value):
    target.opt_value_weight_last = value["weight"]
    target.opt_value_grad_last = value["grad"]
    target.opt_value_weight += value["weight"]
    target.opt_value_grad += value["grad"]
    target.opt_count += 1

def ResetSegment(target):
    target.opt_value_weight = 0
    target.opt_value_grad = 0
    target.opt_count = 0

def GetSegment(target):
    if target.opt_count == 0:
        return (target.opt_value_weight_last, target.opt_value_grad_last), 0, (target.opt_value_weight/1, target.opt_value_grad/1)
    return (target.opt_value_weight_last, target.opt_value_grad_last), target.opt_count, (target.opt_value_weight/target.opt_count, target.opt_value_grad/target.opt_count)


CANDIDATE = ["FB24", "FB16", "FB12"]
THRESHOLD_UP = [0.65, 0.55]
THRESHOLD_DOWN = [0.45, 0.35]
HOLDING = 3

from utils.logger import rCol, tCol, bCol

# Much color. So colorful
def CoLoRiZe(val, format = "%2.4f"):
    # magenta - red - yellow - green - cyan - blue
    tl = [tCol['b'], tCol['bb'], tCol['c'], tCol['g'], tCol['y'], tCol['r'], tCol['r'], tCol['r'], tCol['m'], tCol['m']]
    tc = tl[int(val*len(tl))]
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

    def Initialize(self, net, step = -1, log_dir = "", option = ""):
        Log.Print("=== DYNAMIC OPTIMIZER INITIALIZING ===", elapsed=False, current=False)
        self.layerNames = GetBFLayerNames(net)
        # Print information about detected layers
        Log.Print("Detected Layer:", elapsed=False, current=False)
        for i in self.layerNames:
            PrepareSegment(getattr_(net, i[4:]), i)    
            s = " - " + str(i) + " : " + str(type(getattr_(net, i[4:])))
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

            self.log_file.write(option + "\n")
            # Write Information
            s = "\n"
            self.log_file.write(s)
            # Write Table Information
            s = "Step\t"
            for i in self.layerNames:
                s += i + ".weight\t" + i + ".grad\t"
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
            svv += CoLoRiZeB(layer.bfp_conf.fw_bit) + CoLoRiZe(v[0]) + "/" + CoLoRiZeB(layer.bfp_conf.bwg_bit) + CoLoRiZe(v[1]) + " "
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
            v["grad"] = get_zse(layer.weight.grad, layer.bfp_conf.bwg_bit, layer.bfp_conf.bwg_dim)
            v["weight"] = get_zse(layer.weight, layer.bfp_conf.fw_bit, layer.bfp_conf.fw_dim)
            UpdateSegment(layer, v)

        self.CoLoRpRiNt(net)
        
        if self.optimizeStep != -1 and self.updateCount == self.optimizeStep:
            self.Optimize(net)

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
            s = " - " + str(i) + " : %2.4f/%2.4f "%(a[0],a[1])
            
            if self.optimizeCount > 100:
                if a[0] < 0.4:
                    if layer.bfp_conf.fw_bit == 16:
                        layer.bfp_conf.fw_bit = 8
                    elif layer.bfp_conf.fw_bit == 8:
                        layer.bfp_conf.fw_bit = 4
                elif a[0] > 0.6:
                    if layer.bfp_conf.fw_bit == 4:
                        layer.bfp_conf.fw_bit = 8
                    elif layer.bfp_conf.fw_bit == 8:
                        layer.bfp_conf.fw_bit = 16
                if a[1] < 0.4:
                    if layer.bfp_conf.bwg_bit == 16:
                        layer.bfp_conf.bwg_bit = 8
                    elif layer.bfp_conf.bwg_bit == 8:
                        layer.bfp_conf.bwg_bit = 4
                elif a[1] > 0.6:
                    if layer.bfp_conf.bwg_bit == 4:
                        layer.bfp_conf.bwg_bit = 8
                    elif layer.bfp_conf.bwg_bit == 8:
                        layer.bfp_conf.bwg_bit = 16
            if prev_fw_bit != layer.bfp_conf.fw_bit:
                s += "fw %2d->%2d"%(prev_fw_bit, layer.bfp_conf.fw_bit)
            if prev_bwg_bit != layer.bfp_conf.bwg_bit:
                s += "bwg %2d->%2d"%(prev_bwg_bit, layer.bfp_conf.bwg_bit)
            Log.Print(s, elapsed=False, current=False)

        # Show overall
        sl = str(self.optimizeCount) + "\t"
        sd = str(self.optimizeCount) + "\t"
        for i in self.layerNames:
            layer = getattr_(net, i[4:])
            sl += "%d\t%d\t"%(layer.bfp_conf.fw_bit,layer.bfp_conf.bwg_bit)
            _, _, a = GetSegment(layer)
            sd += "%.4f\t%.4f\t"%(a[0],a[1])
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