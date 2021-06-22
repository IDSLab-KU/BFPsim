import torch
import torch.optim as optim
import torch.nn as nn

from log import Log
from functions import LoadDataset, BFConf, Stat, str2bool, SaveModel, GetNetwork, GetOptimizerScheduler
from train import TrainNetwork
from utils import ZSEAnalyze, SaveNetworkWeights


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

def GetBFLayerConfig(file, model):
    if file == "":
        # Log.Print("bf layer config file not set, FP32 network config is selected.", current=False, elapsed=False)
        # Log.Print("Ignore any additional warnings around setting layers", current=False, elapsed=False)
        conf = dict()
    elif not os.path.exists("./conf_net/"+file+".json"):
        raise FileNotFoundError(file + ".json not exists on ./conf_net/ directory!")
    else:
        with open("./conf_net/"+file+".json","r",encoding="utf-8") as f:
            conf = json.load(f)
        if conf["name"] != model:
            raise ValueError("BF layer configuration file not match with model")
    return conf

# Parse arguments
def ArgumentParse():
    parser = argparse.ArgumentParser()

    # Base mode select
    parser.add_argument("--mode", type=str, default = "train",
        help = "Program execting mode [train, zse-analyze, save-network-weights]")
    parser.add_argument("--cuda", type=str2bool, default=True,
        help = "Using CUDA to compute on GPU [True, False]")

    """Train mode"""
    # Data loader
    parser.add_argument("-d","--dataset", type=str, default = "CIFAR10",
        help = "Dataset to use [CIFAR10, CIFAR100]")
    parser.add_argument("-nw","--num-workers", type=int, default = 4,
        help = "Number of workers to load data")

    # Super setup
    parser.add_argument("-tc", "--train-config-file", type=str, default="",
        help = "Load train config from a file. Option with tag [TC] is configuable with config file. More specifially, see documentation")

    # Model setup
    parser.add_argument("-m","--model", type=str, default = "ResNet18",
        help = "[TC] Model to use [AlexNet, ResNet18, MobileNetv1, DenseNetCifar, VGG11]")
    parser.add_argument("-bf", "--bf-layer-conf-file", type=str, default="",
        help = "Config of the bf setup, if not set, original network will be trained")
    
    # Training setup
    parser.add_argument("--training-epochs", type=int, default = 200,
        help = "[TC] Training epochs")
    parser.add_argument("--loss-boost", type=float, default = 1.0,
        help = "[TC] Loss Boost")
    # parser.add_argument("--initial-lr", type=float, default = 0.1,
    #     help = "Initial learning rate")
    # parser.add_argument("--momentum", type=float, default = 0,
    #     help = "Momentum")

    parser.add_argument("--train-accuracy", type=str2bool, default = False,
        help = "If True, prints train accuracy (slower)")

    # Block setup
    # Printing / Logger / Stat
    parser.add_argument("--save-name", type=str, default = "",
        help = "[TC] Name of the save file")

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
    
    """Network Weight Save / ZSE Evaluation mode"""
    parser.add_argument("--save-file", type=str, default = "",
        help = "Saved checkpoint to load model")
    
    """ZSE Evaluation mode"""
    parser.add_argument("--zt-bf", type=str2bool, default = False,
        help = "[Zero test] If saved file is BF network, set this to true")
    parser.add_argument("--zt-graph-mode", type=str, default="percentage",
        help = "[Zero test] graphing mode [none, percentage, count]")
    parser.add_argument("--zt-print-mode", type=str, default = "sum",
        help = "[Zero test] Print mode [none, sum, format, all]")

    # Parse arguments
    args = parser.parse_args()

    # Save log file location
    
    # Load train config file
    if args.train_config_file == "":
        Log.Print("Train config not set, simple mode activated.", current=False, elapsed=False)
        args.train_config = None
    elif not os.path.exists("./conf_train/"+args.train_config_file+".json"):
        raise FileNotFoundError(args.train_config_file + ".json not exists on ./conf_train/ directory!")
    else:
        with open("./conf_train/"+args.train_config_file+".json", "r", encoding="utf-8") as f:
            args.train_config = json.load(f)

    if args.train_config != None and "model" in args.train_config:
        args.save_name = args.train_config["save-name"]
    elif args.save_name == "":
        args.save_name = str(datetime.now())[:-7].replace("-","").replace(":","").replace(" ","_")
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./saves"):
        os.makedirs("./saves")
    if not os.path.exists("./stats"):
        os.makedirs("./stats")

    args.log_location = "./logs/" + args.save_name + ".log"
    args.save_prefix = "./saves/" + args.save_name
    args.stat_location = "./stats/" + args.save_name + ".stat"

    # Set log
    if args.log:
        Log.SetLogFile(True, args.log_location)
    else:
        Log.SetLogFile(False)
    
    # Load train data
    args.trainset, args.testset, args.classes = LoadDataset(args.dataset)

    if args.train_config != None and "model" in args.train_config:
        args.model = args.train_config["model"]


    # Setting the model
    if args.train_config == None:
        # Parse bf layer conf from file
        args.bf_layer_conf = GetBFLayerConfig(args.bf_layer_conf_file, args.model)
        # Define the network and optimize almost everything
        args.net = GetNetwork(args.model, args.bf_layer_conf, args.classes, args.loss_boost)
    elif "bf-layer-conf-dict" not in args.train_config:
        Log.Print("bf-layer-conf-dict is not set. bf-layer-conf-file will be used.", current=False, elapsed=False)
        if "bf-layer-conf-file" not in args.train_config:
            raise ValueError("bf-layer-conf-file is not set. Please provide at least from bf-layer-conf-file or bf-layer-conf-dict")
        args.bf_layer_conf = GetBFLayerConfig(args.train_config["bf_layer_conf_file"], args.model)
        args.net = GetNetwork(args.model, args.bf_layer_conf, args.classes, args.loss_boost)
    else:
        Log.Print("Training with checkpoints", current=False, elapsed=False)
        args.bf_layer_confs = []
        args.checkpoints = []
        Log.Print("Checkpoints", current=False, elapsed=False)
        for key, value in args.train_config["bf-layer-conf-dict"].items():
            if len(args.checkpoints) > 0 and int(key) <= args.checkpoints[len(args.checkpoints) - 1]:
                ValueError("bf-layer-conf-dict's checkpoint epoch is invalid. %d <= %d"%(int(key), args.checkpoints[len(args.checkpoints) - 1]))
            args.checkpoints.append(int(key))
            b = GetBFLayerConfig(value, args.model)
            args.bf_layer_confs.append(b)
            Log.Print("    %d: Epoch %4d = %s"%(len(args.checkpoints), args.checkpoints[len(args.checkpoints)-1], value), current=False, elapsed=False)
        # Error tracking
        if args.checkpoints[0] != 0:
            raise ValueError("bf-layer-conf-dict's first checkpoint's epoch is not 0")
        # Load the first checkpoint of the model
        args.net = GetNetwork(args.model, args.bf_layer_confs[0], args.classes, args.loss_boost)

    # Set the print interval
    if args.print_train_count == -1:
        args.print_train_count = 5 # Reduced print rate
        
    # Trainloader and Testloader
    args.batch_size_train = 128
    args.batch_size_test = 100
    
    # Critertion, optimizer, scheduler
    args.criterion = nn.CrossEntropyLoss()


    if args.train_config != None and "optimizer-dict" in args.train_config:
        if "0" in args.train_config["optimizer-dict"]:
            args.optimizer, args.scheduler = GetOptimizerScheduler(args.net, args.train_config["optimizer-dict"]["0"])
        else:
            args.optimizer, args.scheduler = GetOptimizerScheduler(args.net)            
    else:
        args.optimizer, args.scheduler = GetOptimizerScheduler(args.net)

    # Training Epochs    
    if args.train_config != None and "training-epochs" in args.train_config:
        args.training_epochs = args.train_config["training-epochs"]
    else:
        args.training_epochs = 200
    
    if args.train_config != None and "loss-boost" in args.train_config:
        args.loss_boost = args.train_config["loss-boost"]

    # Logger, stat, etc
    args.stat_loss_batches = 100

    # Move model to gpu if gpu is available
    if args.cuda:
        args.net.to('cuda')
        # Dataparallel temporary disabled
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
            if arg in ["mode", "dataset", "num_workers", "train_config_file", "model", "bf_layer_config_file", "training_epochs", "save_name", "log_location", "stat_location", "save_prefix", "net", "batch_size_train", "batch_size_test", "criterion", "optimizer", "scheduler"]:
                Log.Print(str(arg) + " : " + str(getattr(args, arg)), current=False, elapsed=False)
        TrainNetwork(args)
    elif args.mode == "zse-analyze":
        Log.Print("Program executed on zse-analyze mode.", current=False, elapsed=False)
        Log.Print("Loaded saved file: {}".format(args.save_file), current=False, elapsed=False)
        Log.Print("Graph mode: {}".format(args.zt_graph_mode), current=False, elapsed=False)
        Log.Print("Print mode: {}".format(args.zt_print_mode), current=False, elapsed=False)
        ZSEAnalyze(args)
    elif args.mode == "save-network-weights":
        Log.Print("Program executed on save-network-weights mode.", current=False, elapsed=False)
        Log.Print("Loaded saved file: {}".format(args.save_file), current=False, elapsed=False)
        SaveNetworkWeights(args)
    elif args.mode == "temp":
        pass
        #for name, param in args.net.named_parameters():
        #     
        #     if param.requires_grad:
        #         Log.Print(name, current=False, elapsed=False)
            # Log.Print(parameter, current=False, elapsed=False)
        # Log.Print("==", current=False, elapsed=False)
    else:
        raise NotImplementedError("Mode not supported : {}".format(args.mode))
