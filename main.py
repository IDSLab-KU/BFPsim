import torch
import torch.optim as optim
import torch.nn as nn

from log import Log
from functions import LoadDataset, BFConf, Stat, str2bool, SaveModel
from train import TrainNetwork
from utils import ZeroTest, SaveData

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

def handler(signum, frame):
    print("Quit by user signal")
    if args != None:
        if args.stat is not None:
            Log.Print("Saving stat object file...")
            args.stat.SaveToFile()
        
        if args.save:
            SaveModel(args, "canceled")
    sys.exit()

# Parse arguments
def ArgumentParse():
    parser = argparse.ArgumentParser()

    # Base mode select
    parser.add_argument("--mode", type=str, default = "train",
        help = "Dataset to use [train, zero-test]")

    """Train mode"""
    # Data loader
    parser.add_argument("-d","--dataset", type=str, default = "CIFAR10",
        help = "Dataset to use [CIFAR10, CIFAR100]")
    parser.add_argument("-nw","--num-workers", type=int, default = 4,
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
        TrainNetwork(args)
    elif args.mode == "zero-test":
        Log.Print("Program executed on zero-test mode.", current=False, elapsed=False)
        Log.Print("Loaded saved file: {}".format(args.save_file), current=False, elapsed=False)
        Log.Print("Graph mode: {}".format(args.zt_graph_mode), current=False, elapsed=False)
        Log.Print("Print mode: {}".format(args.zt_print_mode), current=False, elapsed=False)
        ZeroTest(args)
    elif args.mode == "save-data":
        Log.Print("Program executed on save-data mode.", current=False, elapsed=False)
        Log.Print("Loaded saved file: {}".format(args.save_file), current=False, elapsed=False)
        SaveData(args)
    else:
        raise NotImplementedError("Mode not supported : {}".format(args.mode))
