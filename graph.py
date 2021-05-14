

import os
import numpy as np
import matplotlib.pyplot as plt


class dataValue:
    def __init__(self, name):
        self.name = name
        self.loss = []
        self.acc = []
    
    def AssignLoss(self, loss):
        self.loss = loss    
    def AssignAccuracy(self, acc):
        self.acc = acc    
    def MinLoss(self):
        return min(self.loss)
    def MaxAccuracy(self):
        return max(self.acc)
    def __repr__(self):
        return "%40s/L:%2.8f/A:%2.4f"%(self.name,self.MinLoss(),self.MaxAccuracy())

def ParseFloatList(list):
    res = []
    for val in list:
        try:
            res.append(float(val))
        except:
            pass
    return res


def PlotData(data, mode="Loss", ylim=None, save=True, show=False):
    plt.rcParams["figure.figsize"] = (12,5)
    for d in data:
        label = d.name.split("_")[-1].split(".")[0]
        if mode=="Loss":
            plt.plot(d.loss, label=label)
        elif mode=="Accuracy":
            plt.plot(d.acc, label=label)
    plt.legend()

    title = mode +" of "
    for i in data[0].name.split("_")[:-1]:
        title += i + " "
    title = title[:-1]
    plt.title(title)

    if mode == "Loss":
        plt.xlabel("100 Batches")
    elif mode == "Accuracy":
        plt.xlabel("Epoch")

    plt.ylabel(mode)
    if ylim != None:
        plt.ylim(ylim)
    if save:
        plt.savefig(title.replace(" ","_"))
    if show:
        plt.show()

    plt.clf()


if __name__=="__main__":
    data = []
    # Load All data
    for filename in os.listdir("./stats"):
        with open(os.path.join("./stats", filename), 'r') as f: # open in readonly mode
            # print(filename)
            mode = ""
            d = dataValue(filename)

        
            line = f.readline()
            line = f.readline()
            d.AssignLoss(ParseFloatList(line.split("\t")))
            line = f.readline()
            line = f.readline()
            d.AssignAccuracy(ParseFloatList(line.split("\t")))

            data.append(d)            
            # do your stuff
    
    for d in data:
        print(d)
    PlotData(data, mode="Loss", ylim=(0,2))
    PlotData(data, mode="Accuracy", ylim=(0,1))