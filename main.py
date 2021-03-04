import torch
import torch.optim as optim
import torch.nn as nn

from log import Log
from functions import ArgumentParse

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


def Train():
    running_loss = 0.0
    batch_count = 0
    ptc_target = 1.0 / args.print_train_count
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
                ptc_target += 1/args.print_train_count
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
    
    # Set logger preset
    Log.SetLogFile(True)
    # Parse Arguments and set
    args = ArgumentParse(Log.logFileLocation)
    
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
