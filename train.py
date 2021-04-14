import torch

from log import Log
from functions import SaveModel

def Train(args, epoch_current):
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


def Evaluate(args):
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


def EvaluateTrain(args):
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
def TrainNetwork(args):
    for epoch_current in range(args.training_epochs):
        Train(args, epoch_current)
        Evaluate(args)
        if args.train_accuracy:
            EvaluateTrain(args)
            
        if args.save:
            if args.save_interval != 0 and (epoch_current+1)%args.save_interval == 0:
                SaveModel(args, "%03d"%(epoch_current+1))
    
    Log.Print('Finished Training')

    if args.stat is not None:
        Log.Print("Saving stat object file...")
        args.stat.SaveToFile()

    if args.save:
        SaveModel(args, "finish")
