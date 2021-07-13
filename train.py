import torch
import torch.optim as optim

from utils.logger import Log
from utils.slackBot import slackBot
from functions import SaveModel, GetNetwork, GetOptimizerScheduler

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

        # Boost Loss
        loss *= args.loss_boost

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
            if (i + 1) / len(args.trainloader) >= ptc_target:
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
    
    Log.Print('Test Accuracy: %f, lr: %f' % (correct / total, args.optimizer.param_groups[0]['lr']))
    if args.stat is not None:
        args.stat.AddTestAccuracy(correct / total)
    slackBot.AppendDump("%f "%(correct/total))
        


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
    
    Log.Print('Train Accuracy: %f, lr: %f' % (correct / total, args.optimizer.param_groups[0]['lr']))
    if args.stat is not None:
        args.stat.AddTrainAccuracy(correct / total)
    

# Train the network and evaluate
def TrainNetwork(args):
    Log.Print("========== Starting Training ==========")
    checkpointIndex = 0 # Index of the checkpoint
    slackBot.ResetStartTime()
    for epoch_current in range(args.training_epochs):

        # Change and transfer model
        if args.train_config != None and len(args.checkpoints) > checkpointIndex+1 and epoch_current == args.checkpoints[checkpointIndex+1]:
            checkpointIndex += 1
            s = str(args.checkpoints[checkpointIndex])
            name = args.train_config["bf-layer-conf-dict"][s] if args.train_config["bf-layer-conf-dict"][s] != "" else "None"
            Log.Print('Changing Model bf config to: %s'%(name), elapsed=False, current=False)
            # Save original net
            net_ = args.net
            # Create new net
            args.net = GetNetwork(args.model, args.bf_layer_confs[checkpointIndex], args.classes, args.loss_boost, args.dataset)           
            # Copy state dicts
            args.net.load_state_dict(net_.state_dict())
            # Create new optimizer and emulate step
            if "optimizer-dict" in args.train_config:
                if s in args.train_config["optimizer-dict"]:
                    args.optimizer, args.scheduler = GetOptimizerScheduler(args.net, args.train_config["optimizer-dict"][s])
                else:
                    args.optimizer, args.scheduler = GetOptimizerScheduler(args.net)            
            else:
                args.optimizer, args.scheduler = GetOptimizerScheduler(args.net)
            
            # To gpu
            if args.cuda:
                args.net.to('cuda')
            
        
        Train(args, epoch_current)
        Evaluate(args)
        if args.print_train_accuracy:
            EvaluateTrain(args)
        
        if args.save:
            if args.save_interval != 0 and (epoch_current+1)%args.save_interval == 0:
                SaveModel(args, "%03d"%(epoch_current+1))
        
        if (epoch_current+1) == 1:
            slackBot.SendProgress(float(epoch_current+1)/args.training_epochs, length=0)
        
        # if (epoch_current+1) % 5 == 0:
        #     slackBot.SendProgress(float(epoch_current+1)/args.training_epochs)

    slackBot.SendDump()
    Log.Print("========== Finishing Training ==========")

    if args.stat is not None:
        Log.Print("Saving stat object file...")
        args.stat.SaveToFile()

    if args.save:
        SaveModel(args, "finish")
