import torch
import torch.optim as optim
import torch.nn as nn


from functions import Stat, str2bool, SaveModel, GetNetwork, GetOptimizerScheduler
from train import TrainNetwork

from utils.logger import Log
from utils.generateConfig import GenerateConfig
from utils.ZSEAnalyze import ZSEAnalyze

from dataset import LoadDataset

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
        Log.Print(file + ".json not found, returning empty bf_conf_dict...", current=False, elapsed=False)
        return dict()
        # raise FileNotFoundError(file + ".json not exists on ./conf_net/ directory!")
    else:
        with open("./conf_net/"+file+".json","r",encoding="utf-8") as f:
            conf = json.load(f)
        if conf["name"] != model:
            raise ValueError("BF layer configuration file not match with model")
    return conf

# Parse arguments
def ArgumentParse():
    parser = argparse.ArgumentParser()

    # Global options
    parser.add_argument("--mode", type=str, default="train",
        help = "Program execution mode [train, zseAnalyze, generateConfig]")
    parser.add_argument("--cuda", type=str2bool, default=True,
        help = "True if you want to use cuda")

    # tag:trainConfig
    parser.add_argument("-tc", "--train-config-file", type=str, default="",
        help = "Train config file. Please see documentation about usage")

    # ----------------- configurable by train config file -------------------
    # tag:Save
    parser.add_argument("--save-name", type=str, default = "",
        help = "Name of the saved log file, stat object, save checkpoint")
    parser.add_argument("--log", type=str2bool, default = True,
        help = "Set true to save log file")
    parser.add_argument("--stat", type=str2bool, default = False,
        help = "[Train] Record to stat object?")
    parser.add_argument("--stat-loss-batches", type=int, default = -1,
        help = "[Train] # of batches to calculate average loss. If not set, it will set to count of batches")
    parser.add_argument("--save", type=str2bool, default = False,
        help = "[Train] Set true to save checkpoints")
    parser.add_argument("--save-interval", type=int, default = 0,
        help = "[Train] Checkpoint save interval. 0:only last, rest:interval")
    
    
    # tag:Dataset
    parser.add_argument("-d","--dataset", type=str, default = "CIFAR10",
        help = "Dataset to use [CIFAR10, CIFAR100, ImageNet]")
    parser.add_argument("-dp","--dataset-path", type=str, default = "./data",
        help = "Provide if dataset is prepared, escpecially for ImageNet. Dataset will automatically downloaded on CIFAR-10 and CIFAR-100")
    parser.add_argument("--dataset-pin-memory", type=str2bool, default = True,
        help = "Setting this option will pin dataset to memory, which may boost execution speed.")
    parser.add_argument("--num-workers", type=int, default = 4,
        help = "Recommended value : 4")
    parser.add_argument("--batch-size-train", type=int, default = 128,
        help = ".")
    parser.add_argument("--batch-size-test", type=int, default = 100,
        help = ".")

    # tag:Model
    parser.add_argument("-m","--model", type=str, default = "ResNet18",
        help = "Model [AlexNet, ResNet18, MobileNetv1, DenseNetCifar, VGG16, MLPMixerB16]")
    parser.add_argument("-bf", "--bf-layer-conf-file", type=str, default="",
        help = "Config of the bf setup, if not set, original network will be trained")
    
    # tag:Train
    parser.add_argument("--training-epochs", type=int, default = 200,
        help = ".")
    parser.add_argument("--loss-boost", type=float, default = 1.0,
        help = "Loss boost to each layer [NOT IMPLEMENTED]")

    # tag:Print
    parser.add_argument("--print-train-accuracy", type=str2bool, default = False,
        help = "If True, prints train accuracy (slower)")
    parser.add_argument("--print-train-batch", type=int, default = 0,
        help = "Print progress on # of batches, 0 to disable") # 128 = 391
    parser.add_argument("--print-train-count", type=int, default = 5,
        help = "# of print progress on each epoch, 0 to disable") 

    # ------------- end of configurable by train config file ---------------

    # Tag:zseAnalyze
    parser.add_argument("--save-file", type=str, default = "",
        help = "Saved checkpoint of the model")
    parser.add_argument("--zse-bf", type=str2bool, default = False,
        help = "[zseAnalyze] If saved file is BF network, set this to true")
    parser.add_argument("--zse-graph-mode", type=str, default="percentage",
        help = "[zseAnalyze] Choose graph mode [none, percentage, count]")
    parser.add_argument("--zse-print-mode", type=str, default = "sum",
        help = "[zseAnalyze] Choose print mode [none, sum, format, all]")

    # Parse arguments
    args = parser.parse_args()

    # tag:trainConfig
    if args.train_config_file == "":
        Log.Print("Train Config file not set. Using command line options...", current=False, elapsed=False)
        args.train_config = None
    elif not os.path.exists("./conf_train/"+args.train_config_file+".json"):
        raise FileNotFoundError(args.train_config_file + ".json not exists on ./conf_train/ directory!")
    else:
        with open("./conf_train/"+args.train_config_file+".json", "r", encoding="utf-8") as f:
            args.train_config = json.load(f)

    # tag:Save
    if args.train_config != None and "save-name" in args.train_config:
        args.save_name = args.train_config["save-name"]
    elif args.save_name == "":
        args.save_name = str(datetime.now())[:-7].replace("-","").replace(":","").replace(" ","_")
    if args.train_config != None and "log" in args.train_config:
        args.log = args.train_config["log"]
    if args.train_config != None and "stat" in args.train_config:
        args.stat = args.train_config["stat"]
    if args.train_config != None and "stat-loss-batches" in args.train_config:
        args.stat_loss_batches = args.train_config["stat-loss-batches"]
    if args.train_config != None and "save" in args.train_config:
        args.save = args.train_config["save"]
    if args.train_config != None and "save-interval" in args.train_config:
        args.save_interval = args.train_config["save-interval"]

    # Additional handlers
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./saves"):
        os.makedirs("./saves")
    if not os.path.exists("./stats"):
        os.makedirs("./stats")
    args.log_location = "./logs/" + args.save_name + ".log"
    args.save_prefix = "./saves/" + args.save_name
    args.stat_location = "./stats/" + args.save_name + ".stat"
    # Set the log file
    Log.SetLogFile(True, args.log_location) if args.log else Log.SetLogFile(False)
    args.stat = Stat(args) if args.stat else None
    
    # tag:Dataset
    if args.train_config != None and "dataset" in args.train_config:
        args.dataset = args.train_config["dataset"]
    if args.train_config != None and "dataset-path" in args.train_config:
        args.dataset_path = args.train_config["dataset-path"]
    if args.train_config != None and "dataset-pin-memory" in args.train_config:
        args.dataset_pin_memory = args.train_config["dataset-pin-memory"]
    if args.train_config != None and "num-workers" in args.train_config:
        args.num_workers = args.train_config["num-workers"]
    if args.train_config != None and "batch-size-train" in args.train_config:
        args.batch_size_train = args.train_config["batch-size-train"]
    if args.train_config != None and "batch-size-test" in args.train_config:
        args.batch_size_test = args.train_config["batch-size-test"]
    
    # Load dataset
    args.trainset, args.testset, args.classes, args.trainloader, args.testloader = LoadDataset(args)
    args.batch_count = len(args.trainloader)
    if args.stat_loss_batches == -1:
        args.stat_loss_batches = args.batch_count
    
    # tag:Model
    args.bf_layer_confs = []
    args.checkpoints = []
    if args.train_config != None and "model" in args.train_config:
        args.model = args.train_config["model"]
    if args.train_config == None or args.train_config != None and "bf-layer-conf-file" in args.train_config:
        # using one network
        Log.Print("Train config file not set or bf-layer-conf-file is set on config file.")
        if args.train_config != None and "bf-layer-conf-file" in args.train_config:
            args.bf_layer_conf_file = args.train_config["bf-layer-conf-file"]
        args.bf_layer_conf = GetBFLayerConfig(args.bf_layer_conf_file, args.model)

        args.net = GetNetwork(args.model, args.bf_layer_conf, args.classes, args.loss_boost, args.dataset)
    elif "bf-layer-conf-dict" in args.train_config:
        Log.Print("Training with several network configurations", current=False, elapsed=False)
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
        args.net = GetNetwork(args.model, args.bf_layer_confs[0], args.classes, args.loss_boost, args.dataset)
    else:
        raise ValueError("bf-layer-conf-file is not set. Please provide at least from bf-layer-conf-file or bf-layer-conf-dict")

    # tag:Train
    if args.train_config != None and "training-epochs" in args.train_config:
        args.training_epochs = args.train_config["training-epochs"]
    if args.train_config != None and "loss-boost" in args.train_config:
        args.loss_boost = args.train_config["loss-boost"]
    if args.train_config != None and "print-train-accuracy" in args.train_config:
        args.print_train_accuracy = args.train_config["print-train-accuracy"]

    # Critertion, optimizer, scheduler
    # TODO : Fully customizable scheduler...?
    args.criterion = nn.CrossEntropyLoss()
    if args.train_config != None and "optimizer-dict" in args.train_config:
        if "0" in args.train_config["optimizer-dict"]:
            args.optimizer, args.scheduler = GetOptimizerScheduler(args.net, args.train_config["optimizer-dict"]["0"])
        else:
            args.optimizer, args.scheduler = GetOptimizerScheduler(args.net)            
    else:
        args.optimizer, args.scheduler = GetOptimizerScheduler(args.net)

    # tag:Print
    if args.train_config != None and "print-train-batch" in args.train_config:
        args.print_train_batch = args.train_config["print-train-batch"]
    if args.train_config != None and "print-train-count" in args.train_config:
        args.print_train_count = args.train_config["print-train-count"]
    
    # Move model to gpu if gpu is available
    if args.cuda:
        args.net.to('cuda')
        # Dataparallel temporary disabled
        # args.net = torch.nn.DataParallel(args.net) 


    return args

if __name__ == '__main__':
    # handle exit signal, so it can save on user exit
    signal.signal(signal.SIGINT, handler)
    
    # Parse Arguments and prepare almost everything
    args = ArgumentParse()

    Log.Print("Program executed on {} mode.".format(args.mode), current=False, elapsed=False)
    if args.mode == "train":
        # Network training mode
        for arg in vars(args):
            if arg in ["bf_layer_confs", "checkpoints" "trainset", "testset", "classes", "trainloader", "testloader"] or "zse-" in arg:
                continue
            Log.Print(str(arg) + " : " + str(getattr(args, arg)), current=False, elapsed=False)
        TrainNetwork(args)
    elif args.mode == "zseAnalyze":
        # zse analyze mode
        Log.Print("Loaded saved file: {}".format(args.save_file), current=False, elapsed=False)
        Log.Print("Graph mode: {}".format(args.zse_graph_mode), current=False, elapsed=False)
        Log.Print("Print mode: {}".format(args.zse_print_mode), current=False, elapsed=False)
        ZSEAnalyze(args)
    elif args.mode == "generateConfig":
        # Generating config mode
        GenerateConfig(args)
    elif args.mode == "temp":
        pass
    else:
        raise NotImplementedError("Mode not supported : {}".format(args.mode))
