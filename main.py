import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from blocks import BFLinear

from log import Log
import signal
import sys


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=0)

        self.fc1 = BFLinear(64 * 5 * 5, 1024)
        self.fc2 = BFLinear(1024, 1024)
        self.fc3 = BFLinear(1024, 10)

    def forward(self, x):
                                        #  3x32x32
        x = F.relu(self.conv1(x))       # 16x32x32
        x = self.pool1(x)               # 16x16x16
        x = F.relu(self.conv2(x))       # 32x14x14
        x = self.pool2(x)               # 32x 7x 7
        x = F.relu(self.conv3(x))       # 64x 5x 5
        x = x.view(-1, 64 * 5 * 5)      # 1600
        x = F.relu(self.fc1(x))         # 1024
        x = F.relu(self.fc2(x))         # 1024
        x = self.fc3(x)                 # 10
        return x
    



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



def Train(net, trainloader, testloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    lr = 0.0001

    for epoch in range(5):  # loop over the dataset multiple times

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

            # Manually designed optimizer
            optimizer.step()
            """
            with torch.no_grad():
                net.conv1.weight -= lr * net.conv1.weight.grad
                net.conv2.weight -= lr * net.conv2.weight.grad
                net.conv3.weight -= lr * net.conv3.weight.grad
                net.fc1.weight -= lr * net.fc1.weight.grad
                net.fc2.weight -= lr * net.fc2.weight.grad
                net.fc3.weight -= lr * net.fc3.weight.grad
            """

            # Exit after few iterations to check it's working
            # ExitCounter()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % 2500 == 0:    # print every 2000 mini-batches
                Log.Print('[%d, %5d/%5d] loss: %.3f' %
                    (epoch + 1, i + 1, len(trainloader), running_loss / 2500))
                running_loss = 0.0
                
        Evaluate(net, testloader)
    Log.Print('Finished Training')

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    
    # Set logger preset
    Log.SetLogFile(True)

    # Set transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Prepare Cifar-10 Dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    
    # Define the network
    net = Net()
    Log.Print("Model Summary:%s"%net,current=False, elapsed=False)
    # Train the network
    Train(net, trainloader, testloader)
    






""" save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
"""
