import torch
import torch.optim as optim
import torch.nn as nn

from net import SimpleNet, Resnet18
from log import Log
from functions import LoadDataset

import signal
import sys
import argparse

def str2bool(v):
    if v.lower() in ["true", "t", "1"]: return True
    elif v.lower() in ["false", "f", "0"]: return False
    else: raise argparse.ArgumentTypeError("Not Boolean value")

exitToken = 10
def ExitCounter():
    global exitToken
    exitToken -= 1
    if exitToken == 0:
        sys.exit()

def handler(signum, frame):
    # Log.Exit()
    print('Quit by user', signum)
    sys.exit()


def ArgumentParse(print=True):
    s = "List of the training arguments\n"
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", type=str, default = "CIFAR-10",
        help = "Dataset to use [CIFAR-10, CIFAR-100]")
    parser.add_argument("-m","--model", type=str, default = "Resnet18",
        help = "Model to use [SimpleNet, Resnet18]")

    # Training setup
    parser.add_argument("-bst","--batch-size-training", type=int, default = 4,
        help = "Size of the mini-batch on training")
    parser.add_argument("-bse","--batch-size-evaluation", type=int, default = 4,
        help = "Size of the mini-batch on evaluation")
    parser.add_argument("-e","--training-epochs", type=int, default = 5,
        help = "Number of epochs to train")


    # Printing method
    parser.add_argument("-pti","--print-train-interval", type=int, default = 2500,
        help = "Print interval")
    args = parser.parse_args()

    s += str(args) + "\n"

    if print:
        Log.Print(s, current=False, elapsed=False)
    return args



def Evaluate(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    Log.Print('Accuracy: %f' % (correct / total))



def Train(net, args, trainloader, testloader):
    # Sometimes, there need to optimize optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epoch_train = args.training_epochs

    for epoch_current in range(epoch_train):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # Exit after few iterations to check it's working
            # ExitCounter()

            # print statistics
            running_loss += loss.item()
            if args.print_train_interval != 0 and (i + 1) % args.print_train_interval == 0:    # print every 2000 mini-batches
                Log.Print('[%d/%d, %5d/%5d] loss: %.3f' %
                    (epoch_current + 1, epoch_train, i + 1, len(trainloader), running_loss / 2500))
                running_loss = 0.0
                
        Evaluate(net, args, testloader)
    Log.Print('Finished Training')

if __name__ == '__main__':
    # handle signal
    signal.signal(signal.SIGINT, handler)
    
    # Set logger preset
    Log.SetLogFile(True)
    # Parse Arguments
    args = ArgumentParse()

    # Load dataset
    trainset, testset, classes = LoadDataset(args.dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_training, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_evaluation,shuffle=False, num_workers=2)

    # Define the network
    if args.model == "SimpleNet":
        net = SimpleNet(num_classes = len(classes))
    elif args.model == "Resnet18":
        net = Resnet18(num_classes = len(classes))
    else:
        raise NotImplementedError("Model {} not Implemented".format(args.model))

    # Print the model summary
    Log.Print("Model Summary:%s"%net,current=False, elapsed=False)

    # Train the network
    Train(net, args, trainloader, testloader)
    


""" save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
"""
