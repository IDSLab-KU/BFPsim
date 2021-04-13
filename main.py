import torch
import torch.optim as optim
import torch.nn as nn

from log import Log
from functions import LoadDataset, BFConf, Stat, str2bool

from model.AlexNet import AlexNet
from model.ResNet import ResNet18
from model.DenseNet import DenseNetCifar
from model.MobileNetv1 import MobileNetv1
from model.VGG import VGG16

import signal
import sys

exitToken = 2
def ExitCounter():
    global exitToken
    exitToken -= 1
    if exitToken == 0:
        sys.exit()

import os
import json
import argparse
from datetime import datetime
args = None


def SaveModel(suffix):
    PATH = "%s_%s.model"%(args.save_prefix,suffix)
    Log.Print("Saving model file as %s"%PATH)
    torch.save(args.net.state_dict(), PATH)


def handler(signum, frame):
    print("Quit by user signal")
    if args != None:
        if args.stat is not None:
            Log.Print("Saving stat object file...")
            args.stat.SaveToFile()
        
        if args.save:
            SaveModel("canceled")
    sys.exit()

# Parse arguments
def ArgumentParse():
    parser = argparse.ArgumentParser()

    # Base mode select
    parser.add_argument("--mode", type=str, default = "train",
        help = "Dataset to use [train, zero-test]")

    """Train mode"""
    # Data loader
    parser.add_argument("-d","--dataset", type=str, default = "CIFAR-10",
        help = "Dataset to use [CIFAR-10, CIFAR-100]")
    parser.add_argument("-nw","--num-workers", type=int, default = 8,
        help = "Number of workers to load data")

    # Model setup
    parser.add_argument("-m","--model", type=str, default = "ResNet18",
        help = "Model to use [AlexNet, ResNet18, MobileNetv1, DenseNetCifar]")
        
    parser.add_argument("-bf", "--bf-layer-conf-file", type=str, default="",
        help = "Config of the bf setup, if not set, original network will be trained")
    parser.add_argument("--cuda", type=str2bool, default=True,
        help = "Using CUDA to compute on GPU [True False]")
    
    # Training setup
    parser.add_argument("--training-epochs", type=int, default = 0,
        help = "[OVERRIDE] If larger than 0, epoch is set to this value")
    parser.add_argument("--initial-lr", type=float, default = 0,
        help = "[OVERRIDE] If larger than 0, initial lr is set to this value")
    parser.add_argument("--momentum", type=float, default = 0,
        help = "[OVERRIDE] If larger than 0, momentum is set to this value")
    parser.add_argument("--train-accuracy", type=str2bool, default = False,
        help = "If True, prints train accuracy (slower)")


    # Block setup
    # Printing / Logger / Stat
    parser.add_argument("--save-name", type=str, default = "",
        help = "Name of the save file")

    parser.add_argument("--log", type=str2bool, default = True,
        help = "Record to log object?")
    parser.add_argument("--print-train-batch", type=int, default = 0,
        help = "Print info on # of batches, 0 to disable") # 128 = 391
    parser.add_argument("--print-train-count", type=int, default = -1,
        help = "How many print on each epoch, 0 to disable") # 128 = 391

    parser.add_argument("--stat", type=str2bool, default = False,
        help = "Record to stat object?")
    parser.add_argument("--stat-loss-batches", type=int, default = 0,
        help = "[OVERRIDE] Average batches to calculate running loss on stat object")

    parser.add_argument("--save", type=str2bool, default = False,
        help = "Save best model's weight")
    parser.add_argument("--save-interval", type=int, default = 0,
        help = "Save interval, 0:last, rest:interval")
    
    """Zero test mode"""

    parser.add_argument("--save-file", type=str, default = "",
        help = "Saved checkpoint to load model")
    parser.add_argument("--zt-bf", type=str2bool, default = False,
        help = "[Zero test] If saved file is BF network, set this to true")
    parser.add_argument("--zt-graph-mode", type=str, default="percentage",
        help = "[Zero test] graphing mode [none, percentage, count]")
    parser.add_argument("--zt-print-mode", type=str, default = "sum",
        help = "[Zero test] Print mode [none, sum, format, all]")
    # Parse arguments
    args = parser.parse_args()

    # Save log file location
    if args.save_name == "":
        args.save_name = str(datetime.now())[:-7].replace("-","").replace(":","").replace(" ","_")
    """ should create folders by user, not docker.
    It's okay if docker is executed with user mode, which with argument --user "$(id -u):$(id -g)"
    execute preload.sh file to pre-create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./saves"):
        os.makedirs("./saves")
    if not os.path.exists("./stats"):
        os.makedirs("./stats")
    """
    args.log_location = "./logs/" + args.save_name + ".log"
    # args.save_prefix = "./logs/" + args.save_name
    # args.stat_location = "./logs/" + args.save_name + ".stat"
    args.save_prefix = "./saves/" + args.save_name
    args.stat_location = "./stats/" + args.save_name + ".stat"
    if args.log:
        Log.SetLogFile(True, args.log_location)
    else:
        Log.SetLogFile(False)
        
    # Load train data
    args.trainset, args.testset, args.classes = LoadDataset(args.dataset)

    # Parse bf layer conf from file
    if args.bf_layer_conf_file == "":
        Log.Print("bf layer config file not set, original network will be trained.", current=False, elapsed=False)
        Log.Print("Ignore any additional warnings around setting layers", current=False, elapsed=False)
        args.bf_layer_conf = dict()
    elif not os.path.exists("./conf/"+args.bf_layer_conf_file+".json"):
        raise FileNotFoundError(args.bf_layer_conf_file + ".json not exists on ./conf/ directory!")
    else:
        f = open("./conf/"+args.bf_layer_conf_file+".json","r",encoding="utf-8")

        args.bf_layer_conf = json.load(f)
        if args.bf_layer_conf["name"] != args.model:
            raise ValueError("BF layer conf is not match with model")
    

    if args.print_train_count == -1:
        args.print_train_count = 5 # Reduced print rate
        
    # Define the network and optimize almost everything
    if args.model == "AlexNet":
        args.net = AlexNet(args.bf_layer_conf, len(args.classes))
    elif args.model == "ResNet18":
        args.net = ResNet18(args.bf_layer_conf, len(args.classes))
    elif args.model == "VGG16":
        args.net = VGG16(args.bf_layer_conf, len(args.classes))
    elif args.model == "MobileNetv1":
        args.net = MobileNetv1(args.bf_layer_conf, len(args.classes))
    elif args.model == "DenseNetCifar":
        args.net = DenseNetCifar(args.bf_layer_conf, len(args.classes))
    else:
        raise NotImplementedError("Model {} not Implemented".format(args.model))

    # Trainloader and Testloader
    args.batch_size_train = 128
    args.batch_size_test = 100
    
    # Critertion, optimizer, scheduler
    args.criterion = nn.CrossEntropyLoss()
    args.optimizer = optim.SGD(args.net.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    args.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(args.optimizer, T_max=200)

    # Training Epochs
    if args.training_epochs == 0:
        args.training_epochs = 200
    
    # Logger, stat, etc
    args.stat_loss_batches = 100

    # Move model to gpu if gpu is available
    if args.cuda:
        args.net.to('cuda')
        # Temporally disabled because of JAX
        # args.net = torch.nn.DataParallel(args.net) 

    # Testloader and Trainloader
    args.trainloader = torch.utils.data.DataLoader(args.trainset,
        batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    args.testloader = torch.utils.data.DataLoader(args.testset,
        batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)

    # Count of the mini-batches
    args.batch_count = len(args.trainloader)

    # Stat object
    if args.stat:
        args.stat = Stat(args)
    else:
        args.stat = None

    return args

def Train(epoch_current):
    running_loss = 0.0
    batch_count = 0
    ptc_count = 1
    ptc_target = ptc_count / args.print_train_count
    for i, data in enumerate(args.trainloader, 0):
        inputs, labels = data
        
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        args.optimizer.zero_grad()

        outputs = args.net(inputs)
        loss = args.criterion(outputs, labels)
        loss.backward()

        args.optimizer.step()
        running_loss += loss.item()

        # Print the running loss
        pF = False
        batch_count += 1
        if args.print_train_batch != 0:
            if (i + 1) % args.print_train_batch == 0 or (i + 1) == len(args.trainloader):
                pF = True
        elif args.print_train_count != 0:
            if (i+1) / len(args.trainloader) >= ptc_target:
                pF = True
                ptc_count += 1
                ptc_target = ptc_count/args.print_train_count
        if pF:
            Log.Print('[%d/%d, %5d/%5d] loss: %.3f' %
                (epoch_current + 1, args.training_epochs,
                i + 1, len(args.trainloader),
                running_loss / batch_count))
            running_loss = 0.0
            batch_count = 0

        # Record to stat
        if args.stat is not None:
            args.stat.AddLoss(loss.item())

    if args.scheduler != None:
        args.scheduler.step()


def Evaluate():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in args.testloader:
            images, labels = data
            if args.cuda:
                images = images.cuda() # Using GPU
                labels = labels.cuda() # Using GPU
            outputs = args.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    Log.Print('Test Accuracy: %f' % (correct / total))
    if args.stat is not None:
        args.stat.AddTestAccuracy(correct / total)


def EvaluateTrain():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in args.trainloader:
            images, labels = data
            if args.cuda:
                images = images.cuda() # Using GPU
                labels = labels.cuda() # Using GPU
            outputs = args.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    Log.Print('Train Accuracy: %f' % (correct / total))
    if args.stat is not None:
        args.stat.AddTrainAccuracy(correct / total)

# Train the network and evaluate
def TrainNetwork():
    for epoch_current in range(args.training_epochs):
        Train(epoch_current)
        Evaluate()
        if args.train_accuracy:
            EvaluateTrain()
            
        if args.save:
            if args.save_interval != 0 and (epoch_current+1)%args.save_interval == 0:
                SaveModel("%03d"%(epoch_current+1))
    
    Log.Print('Finished Training')

    if args.stat is not None:
        Log.Print("Saving stat object file...")
        args.stat.SaveToFile()

    if args.save:
        SaveModel("finish")


from blockfunc import GetZeroSettingError
import numpy as np
from functions import SaveStackedGraph

def ZeroTest():

    Log.SetPrintCurrentTime(False)
    Log.SetPrintElapsedTime(False)

    parameters = torch.load(args.save_file).items()
    
    layer_count = 0
    layer_list = []
    layer_list_short = []
    Log.Print("List of weight data's name")
    for name, param in parameters:
        n = name.split(".")
        # Normal model name convention
        if args.model == "ResNet18" and args.zt_bf == False:
            condition = "layer" in n[1] and "conv" in n[3] and "weight" in n[4]
            if condition:
                short_name = n[1]+"_"+n[2]+"_"+n[3]
        elif args.model == "ResNet18" and args.zt_bf == True:
            condition = "layer" in n[0] and "conv" in n[2] and "weight" in n[3]
            if condition:
                short_name = n[0]+"_"+n[1]+"_"+n[2]
        else:
            raise NotImplementedError("model and layer conf not implemented")

        if condition:
            Log.Print("{}({})".format(name, param.size()))
            layer_list.append(name)
            layer_list_short.append(short_name)
            layer_count += 1
        
    Log.Print("")

    # for bits in [4]:
    #     for g_size in [36]:
    for bits in [4, 5, 6, 7, 8]:
        for g_size in [36, 54, 72]:
            stat_data = np.zeros((layer_count, bits+1),dtype=np.int32)
            Log.Print("Mantissa bits {}, Group size {}".format(bits, g_size))
            res = np.zeros(bits+1, dtype=np.int32)
            ind = 0
            for name, param in parameters:
                if name in layer_list:
                    d = GetZeroSettingError(param.detach(), bits, g_size, 0)
                    res += d
                    stat_data[ind] = d
                    ind += 1

                    if args.zt_print_mode == "format":
                        for i in d:
                            Log.Print("{}".format(i),end="\t")
                        Log.Print("")
                    if args.zt_print_mode == "all":
                        Log.Print("[{:7.3f}%]{:8d}/{:8d}, {}".format(d[-1]/(d.sum())*100,d[-1],d.sum(),d))
            # Save figures
            if args.zt_graph_mode != "none":
                SaveStackedGraph(layer_list_short, np.flip(stat_data.transpose(),axis=0),
                        mode=args.zt_graph_mode,
                        title="{}, Bit={}, Group={}".format(args.model, bits, g_size),
                        save="{}_{}_{}".format(args.model, bits, g_size))
            if args.zt_print_mode in ["sum", "format", "all"]:
                Log.Print("[{:7.3f}%]{:10d}/{:10d}, {}".format(res[-1]/(res.sum())*100,res[-1],res.sum(),res))            
                Log.Print("")
    # print(args.net)



if __name__ == '__main__':
    # handle signal
    signal.signal(signal.SIGINT, handler)
    
    # Parse Arguments and set
    args = ArgumentParse()

    # Print the model summary and arguments
    # Log.Print("List of the program arguments\n" + str(args) + "\n", current=False, elapsed=False)

    if args.mode == "train":
        for arg in vars(args):
            if arg in ["trainloader", "testloader", "bf_layer_conf", "classes", "testset", "trainset", "stat_loss_batches", "batch_count", "cuda", "log_file_location"]:
                continue
            elif getattr(args,arg) == 0 and arg in ["initial_lr", "momentum", "print_train_batch", "print_train_count"]:
                continue
            else:
                Log.Print(str(arg) + " : " + str(getattr(args, arg)), current=False, elapsed=False)
        TrainNetwork()
    elif args.mode == "zero-test":
        Log.Print("Program executed on zero-test mode.", current=False, elapsed=False)
        Log.Print("Loaded saved file: {}".format(args.save_file), current=False, elapsed=False)
        Log.Print("Graph mode: {}".format(args.zt_graph_mode), current=False, elapsed=False)
        Log.Print("Print mode: {}".format(args.zt_print_mode), current=False, elapsed=False)
        ZeroTest()
        
