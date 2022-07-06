import torch
import torch.optim as optim

from utils.logger import Log
from utils.slackBot import slackBot
from utils.statManager import statManager
from utils.save import SaveModel
from train.network import GetNetwork, GetOptimizer, GetScheduler
from bfp.functions import LoadBFPDictFromFile
from utils.dynamic import DO

def TrainMixed(args, epoch_current):
    running_loss = 0.0
    batch_count = 0
    ptc_count = 1
    ptc_target = ptc_count / args.print_train_count


    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i, data in enumerate(args.trainloader, 0):
            inputs, labels = data
            
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            args.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                outputs = args.net(inputs)
                assert outputs.dtype is torch.float16

                loss = args.criterion(outputs, labels)
                assert loss.dtype is torch.float32

            # loss.backward()
            args.scaler.scale(loss).backward()
            args.scaler.step(args.optimizer)
            args.scaler.update()

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

def Train(args, epoch_current):
    running_loss = 0.0
    batch_count = 0
    ptc_count = 1
    ptc_target = ptc_count / args.print_train_count

    # DO.FlatModel(args.net)

    grad_avg = 0
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i, data in enumerate(args.trainloader, 0):
    
        inputs, labels = data
        
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        args.optimizer.zero_grad()
        # with torch.cuda.amp.autocast(enabled=True):

        outputs = args.net(inputs)
        #     assert outputs.dtype is torch.float16

        loss = args.criterion(outputs, labels)
        #     assert loss.dtype is torch.float32

        loss.backward()
        # args.scaler.scale(loss).backward()
        # args.scaler.step(args.optimizer)
        # args.scaler.update()

        running_loss += loss.item()

        if args.do != "":
            DO.Update(args.net)

        args.optimizer.step()
        args.optimizer.zero_grad()

        
        
        if args.warmup:
            if epoch_current < args.warmup_epoch:
                args.scheduler_warmup.step()
        # Print and record the running loss
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

            args.writer.add_scalar('training loss',
                    running_loss / batch_count,
                    epoch_current * len(args.trainloader) + i)
            # statManager.AddData("training loss", running_loss / batch_count)
            running_loss = 0.0
            batch_count = 0
    
    

"""
Accuracy Code from pytorch example
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
def Accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def Evaluate(args, mode = "test"):
    if mode == "test":
        loader = args.testloader
    elif mode == "train":
        loader = args.trainloader
    else:
        raise ValueError("Mode not supported")
    top1, top3, top5, total = 0, 0, 0, 0
    args.net.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = args.net(images)
            loss = args.criterion(output, target)
            acc1, acc3, acc5 = Accuracy(output, target, topk=(1, 3, 5))
            top1 += acc1[0] * images.size(0)
            top3 += acc3[0] * images.size(0)
            top5 += acc5[0] * images.size(0)
            total += images.size(0)
    return (top1/total).cpu().item(), (top3/total).cpu().item(), (top5/total).cpu().item()


# Train the network and evaluate
def TrainNetwork(args):
    Log.Print("========== Starting Training ==========")
    slackBot.ResetStartTime()

    if args.do != "":
        DO.Initialize(args.net, len(args.trainloader), args.save_prefix, args.do)
        DO.CoLoR = args.do_color

    # args.scaler = torch.cuda.amp.GradScaler() # FP16 Mixed Precision

    for epoch_current in range(args.start_epoch, args.training_epochs):

        

        # Change and transfer model
        if epoch_current != args.start_epoch and str(epoch_current) in args.bfp_layer_conf_dict:
            Log.Print("Changing Model bfp config to: %s"%args.bfp_layer_conf_dict[str(epoch_current)], elapsed=False, current=False)
            net_ = args.net
            args.net = GetNetwork(args.dataset, args.model, args.num_classes, LoadBFPDictFromFile(args.bfp_layer_conf_dict[str(args.start_epoch)]))
            args.net.load_state_dict(net_.state_dict())
            args.net.eval()

            # Load Optimizer, Scheduler, stuffs
            args.optimizer = GetOptimizer(args, str(epoch_current))
            args.scheduler = GetScheduler(args, str(epoch_current))
            if args.cuda:
                args.net.to('cuda')
        

        # Train the net
        Train(args, epoch_current)
        # Evaluate the net
        t1, t3, t5 = Evaluate(args)
        
        args.writer.add_scalar('top1 accuracy', t1, epoch_current)
        # args.writer.add_scalar('top3 accuracy', t3, epoch_current)
        # args.writer.add_scalar('top5 accuracy', t5, epoch_current)

        # statManager.AddData("top1test", t1)
        # statManager.AddData("top3test", t3)
        # statManager.AddData("top5test", t5)
        Log.Print('[%d/%d], TestAcc(t1):%7.3f, lr:%f' % (epoch_current+1, args.training_epochs, t1, args.optimizer.param_groups[0]['lr']))

        if args.scheduler != None:
            args.scheduler.step()

        # Save the model
        if args.save:
            if args.save_interval != 0 and (epoch_current+1)%args.save_interval == 0:
                SaveModel(args, "%03d"%(epoch_current+1))

        # Send progress to printing expected time
        if epoch_current == args.start_epoch:
            slackBot.SendProgress(float(epoch_current+1)/args.training_epochs, length=0)

        # Calculate the zse of the layers and replace model if needed

        # Optional Progress Sending        
        # if (epoch_current+1) % 5 == 0:
        #     slackBot.SendProgress(float(epoch_current+1)/args.training_epochs)

    Log.Print("========== Finished Training ==========")
    
    # Sending remaining dumps
    slackBot.SendDump()
    
    # if args.stat:
    #     Log.Print("Saving stat object file...")
        # statManager.SaveToFile(args.stat_location)

    if args.save:
        SaveModel(args, "finish")
