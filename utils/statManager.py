import numpy as np

class Stat():
    def __init__(self, args):
        self.loss = []
        self.testAccuracy = []
        self.trainAccuracy = []
        self.running_loss = 0.0
        self.loss_count = 0
        self.loss_batches = args.stat_loss_batches
        self.file_location = args.stat_location

    def AddLoss(self, v):
        self.running_loss += v
        self.loss_count += 1
        if self.loss_count == self.loss_batches:
            self.loss.append(self.running_loss / self.loss_batches)
            self.loss_count = 0
            self.running_loss = 0.0
    
    def AddTestAccuracy(self, v):
        self.testAccuracy.append(v)

    def AddTrainAccuracy(self, v):
        self.trainAccuracy.append(v)

    def SaveToFile(self):
        if self.loss_count != 0:
            self.loss.append(self.running_loss / self.loss_batches)
        
        f = open(self.file_location, mode="w+", newline='', encoding='utf-8')
        f.write(">Average Loss per {} batches\n".format(self.loss_batches))
        for i in self.loss:
            f.write(str(i)+"\t")
        f.write("\n")
        f.write("> Test Accuracy\n")
        for i in self.testAccuracy:
            f.write(str(i)+"\t")
        f.write("\n")
        if len(self.trainAccuracy) > 0:
            f.write("> Train Accuracy\n")
            for i in self.trainAccuracy:
                f.write(str(i)+"\t")
            f.write("\n")

class statManager_():
    def __init__(self):
        self.data = dict()
    
    def GetMeterListStr(self):
        if not self.data:
            return "Empty"
        s = "{"
        for i in self.data.keys():
            s +="%s, "%(i)
        s = s[:-2] + "}"
        return s

    def AddMeter(self, name, type=np.float32):
        self.data[name] = np.empty(0, dtype=type)

    def AddData(self, name, value):
        try:
            self.data[name] = np.insert(self.data[name], self.data[name].size, value)
        except:
            print("statManager ERROR: input data's dtype not matches. input:%s != data:%s"%(str(type(value)), str(self.data[name].dtype)))


    def GetLength(self, name):
        if name not in self.data:
            print("statManage ERROR: meter name is not correct. Meters:%s"%(self.GetMeterListStr()))
            return
        return self.data[name].size

    # Get meter information
    def GetMeterString(self, name, recent=-1, fmt="", delim=","):
        return self.GetMeterInfo(name, "string", recent=recent, fmt=fmt, delim=delim)

    def GetMeter(self, name, recent=-1):
        return self.GetMeterInfo(name, "meter", recent=recent)

    def GetAverage(self, name, recent=-1):
        return self.GetMeterInfo(name, "average", recent=recent)

    def GetMax(self, name, recent=-1):
        return self.GetMeterInfo(name, "max", recent=recent)

    def GetMin(self, name, recent=-1):
        return self.GetMeterInfo(name, "min", recent=recent)

    def GetLatest(self, name):
        return self.GetMeterInfo(name, "latest")

    def GetMeterInfo(self, name, mode, recent=-1, fmt="", delim=","):
        if name not in self.data:
            print("statManager ERROR: meter name is not correct. Meters:%s"%(self.GetMeterListStr()))
            return
        if recent > 0:
            if self.data[name].size() < recent:
                print("statManager WARNING: recent range is smaller than meter's length. Returning result of full array... (len:%d, recent:%d)"%(self.data[name].size(), recent))
                v = self.data[name]
            else:
                v = self.data[name][-recent:]
        else:
            v = self.data[name]
        
        if v.size == 0:
            print("statManager WARNING: target is empty array, returning 0.")
            return 0.0
        if mode == "average":
            return np.average(v)
        elif mode == "max":
            return np.max(v)
        elif mode == "min":
            return np.min(v)
        elif mode == "meter":
            return v
        elif mode == "latest":
            return v[-1]
        elif mode == "string":
            s = ""
            for i in v:
                if fmt != "":
                    s += ("{:"+fmt+"}").format(i) + delim
                else:
                    s += "%f,"%i
            return s[:-len(delim)]

    def __repr__(self):
        return str(self)

    def __str__(self):
        s ="   Name   |Count|   Avg   |   Max   |   Min   | Latest  \n"
        for k in self.data.keys():
            s += "%10s|%5d|%9.5f|%9.5f|%9.5f|%9.5f\n"%(k, self.GetLength(k), self.GetAverage(k), self.GetMax(k), self.GetMin(k), self.GetLatest(k))
        return s
            
    def SaveToFile(self, save_dir, fmt="", delim=","):
        f = open(save_dir, mode="w+", newline='', encoding='utf-8')
        f.write(str(self))
        for k in self.data.keys():
            s = "%s\n"%k
            s += str(self.GetMeterString(k, fmt=fmt,delim=delim)) + "\n"
            f.write(s)
        f.close()
    

statManager = statManager_()


print(statManager.AddMeter("top1"))
statManager.AddData("top1", 12.5)
print(statManager.GetMax("top1"))
# print(statManager.GetMeterString("top1", fmt="8.3f",delim=","))
print(statManager)
# statManager.SaveToFile("./save.tmp")