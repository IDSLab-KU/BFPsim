import torch
import torch.optim as optim
import torch.nn as nn

from log import Log
from functions import LoadDataset, Str2Tuple, BFConf, Stat, str2bool
from net import SimpleNet, ResNet18, BFSimpleNet, BFResNet18

import signal
import sys

exitToken = 2
def ExitCounter():
    global exitToken
    exitToken -= 1
    if exitToken == 0:
        sys.exit()

args = None

def handler(signum, frame):
    print("Quit by user signal, saving stat object file...")
    if args.stat != None:
        args.stat.SaveToFile()
    
    if args.save:
        PATH = args.log_file_location[:-4] + ".model"
        torch.save(args.net.state_dict(), PATH)
    sys.exit()



import os
import json

# Argument parsing
import argparse

# Parse arguments
def ArgumentParse(logfileStr):
    parser = argparse.ArgumentParser()
    # Data loader
    parser.add_argument("-d","--dataset", type=str, default = "CIFAR-10",
        help = "Dataset to use [CIFAR-10, CIFAR-100]")
    parser.add_argument("-nw","--num-workers", type=int, default = 8,
        help = "Number of workers to load data")

    # Model setup
    parser.add_argument("-m","--model", type=str, default = "Resnet18",
        help = "Model to use [SimpleNet, Resnet18]")
        
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


    # Block setup

    # Printing / Logger / Stat
    parser.add_argument("--log", type=str2bool, default = True,
        help = "Record to log object?")
    parser.add_argument("--print-train-batch", type=int, default = 0,
        help = "Print info on # of batches, 0 to disable") # 128 = 391
    parser.add_argument("--print-train-count", type=int, default = 5,
        help = "How many print on each epoch, 0 to disable") # 128 = 391
    parser.add_argument("--stat", type=str2bool, default = False,
        help = "Record to stat object?")
    parser.add_argument("--stat-loss-batches", type=int, default = 0,
        help = "[OVERRIDE] Average batches to calculate running loss on stat object")

    parser.add_argument("--save", type=str2bool, default = False,
        help = "Save best model's weight")
    
    # Parse arguments
    args = parser.parse_args()

    # Save log file location
    args.log_file_location = logfileStr
    
    # Load train data
    args.trainset, args.testset, args.classes = LoadDataset(args.dataset)

    # Parse bf layer conf from file
    if args.bf_layer_conf_file == "":
        print("bf layer confing file not set, original network will be trained.")
        args.bf_layer_conf = None
    elif not os.path.exists("./conf/"+args.bf_layer_conf_file+".json"):
        raise FileNotFoundError(args.bf_layer_conf_file + ".json not exists on ./conf/ directory!")
    else:
        f = open("./conf/"+args.bf_layer_conf_file+".json","r",encoding="utf-8")

        args.bf_layer_conf = json.load(f)
        if args.bf_layer_conf["name"] != args.model:
            raise ValueError("BF layer conf is not match with model")
    
    # Define the network and optimize almost everything
    # Simplenet, 3 convs and 3 fc layers
    if args.model == "SimpleNet":
        # Network construction
        if args.bf_layer_conf is not None:
            args.net = BFSimpleNet(args.bf_layer_conf, len(args.classes))
        else:
            args.net = SimpleNet(len(args.classes))

        # Trainloader and Testloader
        args.batch_size_train = 4
        args.batch_size_test = 100
        
        # Critertion, optimizer, scheduler
        args.criterion = nn.CrossEntropyLoss()
        args.optimizer = optim.SGD(args.net.parameters(), lr=0.001, momentum=0.9)
        args.scheduler = None

        # Training Epochs
        if args.training_epochs == 0:
            args.training_epochs = 5
        
        # Logger, stat, etc
        args.stat_loss_batches = 1000
        # args.print_train_batch = 2500

    # Resnet18, 3 convs and 3 fc layers
    elif args.model == "Resnet18":
        # Network construction
        if args.bf_layer_conf is not None:
            args.net = BFResNet18(args.bf_layer_conf, len(args.classes))
        else:
            args.net = ResNet18(len(args.classes))

        # Trainloader and Testloader
        args.batch_size_train = 128
        args.batch_size_test = 100
        
        # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
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
    else:
        raise NotImplementedError("Model {} not Implemented".format(args.model))


    # Testloader and Trainloader
    args.trainloader = torch.utils.data.DataLoader(args.trainset,
        batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    args.testloader = torch.utils.data.DataLoader(args.testset,
        batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers)

    # Move model to gpu if gpu is available
    if args.cuda:
        args.net.to('cuda')
        # Temporally disabled because of JAX
        # args.net = torch.nn.DataParallel(args.net) 

    # Count of the mini-batches
    args.batch_count = len(args.trainloader)

    # Stat object
    if args.stat:
        args.stat = Stat(args)
    else:
        args.stat = None

    return args



def Train():
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
        args.stat.AddAccuracy(correct / total)

if __name__ == '__main__':
    # handle signal
    signal.signal(signal.SIGINT, handler)
    
    # Set logger preset to generate log file location
    Log.SetLogFile(True)
    # Parse Arguments and set
    args = ArgumentParse(Log.logFileLocation)

    if args.log:
        Log.SetLogFile(True)
    else:
        Log.SetLogFile(False)
    
    # Print the model summary and arguments
    
    for arg in vars(args):
        if arg in ["trainloader", "testloader", "bf_layer_conf", "classes", "testset", "trainset", "stat", "stat_loss_batches", "batch_count", "cuda", "log_file_location"]:
            continue
        elif getattr(args,arg) == 0 and arg in ["initial_lr", "momentum", "print_train_batch", "print_train_count"]:
            continue
        else:
            Log.Print(str(arg) + " : " + str(getattr(args, arg)), current=False, elapsed=False)
    # Log.Print("List of the program arguments\n" + str(args) + "\n", current=False, elapsed=False)

    # Train the network
    for epoch_current in range(args.training_epochs):
        Train()
        Evaluate()
    Log.Print('Finished Training')

    if args.stat is not None:
        args.stat.SaveToFile()

    if args.save:
        PATH = args.log_file_location[:-4] + ".model"
        torch.save(args.net.state_dict(), PATH)


""" save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
"""
