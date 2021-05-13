# utils.py

import torch

from log import Log
from blockfunc import GetZeroSettingError
import numpy as np
from functions import SaveStackedGraph



def SaveNetworkWeights(args):
    Log.SetPrintCurrentTime(False)
    Log.SetPrintElapsedTime(False)
    
    args.net.load_state_dict(torch.load(args.save_file))
    # Forward for 1 mini batches
    for i, data in enumerate(args.trainloader, 0):
        inputs, labels = data
        
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # args.optimizer.zero_grad()

        outputs = args.net(inputs)
        # loss = args.criterion(outputs, labels)
        # loss.backward()
        break
    Log.Print("Saved Complete!")

def ZSEAnalyze_(args, bits, g_size):
    parameters = torch.load(args.save_file).items()
    stat_data = np.zeros((args.layer_count, bits+1),dtype=np.int32)
    Log.Print("Mantissa bits {}, Group size {}".format(bits, g_size))
    res = np.zeros(bits+1, dtype=np.int32)
    ind = 0
    for name, param in parameters:
        if name in args.layer_list:
            d = GetZeroSettingError(param.detach(), bits, g_size, 0)
            res += d
            stat_data[ind] = d
            ind += 1

            if args.zt_print_mode == "format":
                for i in d:
                    Log.Print("{}".format(i),end="\t")
                Log.Print("")
            if args.zt_print_mode == "all":
                Log.Print("[{:7.3f}%]{:8d}/{:8d}, {}".format(d[-1]/(d.sum())*100,d[-1],d.sum(),d))
    # Save figures
    if args.zt_graph_mode != "none":
        SaveStackedGraph(args.layer_list_short, np.flip(stat_data.transpose(),axis=0),
                mode=args.zt_graph_mode,
                title="{}, Bit={}, Group={}".format(args.model, bits, g_size),
                save="{}_{}_{}".format(args.model, bits, g_size))
    if args.zt_print_mode in ["sum", "format", "all"]:
        Log.Print("[{:7.3f}%]{:10d}/{:10d}, {}".format(res[-1]/(res.sum())*100,res[-1],res.sum(),res))            
        Log.Print("")


def ZSEAnalyze(args):
    Log.SetPrintCurrentTime(False)
    Log.SetPrintElapsedTime(False)

    parameters = torch.load(args.save_file).items()
    
    args.layer_count = 0
    args.layer_list = []
    args.layer_list_short = []
    Log.Print("List of weight data's name")
    for name, param in parameters:
        n = name.split(".")
        condition = False
        # Normal model name convention
        if args.model == "ResNet18":
            condition = "weight" in n
            if "bn1" in n or "bn2" in n or "linear" in n or "shortcut" in n:
                condition=False
            # Log.Print(str(n))
            # condition = "layer" in n[0] and "conv" in n[2] and "weight" in n[3]
            if condition:
                short_name = "_".join(n[:-1])
        else:
            raise NotImplementedError("model and layer conf not implemented")

        if condition:
            Log.Print("{}({})".format(name, param.size()))
            args.layer_list.append(name)
            args.layer_list_short.append(short_name)
            args.layer_count += 1
        
    Log.Print("")
    # for bits in [4]:
    #     for g_size in [36]:
    # for bits in [4, 5, 6, 7, 8]:
    #     for g_size in [36, 54, 72]:
    ZSEAnalyze_(args, 4, 36)
    ZSEAnalyze_(args, 4, 144)
    ZSEAnalyze_(args, 8, 54)
    ZSEAnalyze_(args, 8, 216)
    ZSEAnalyze_(args, 16, 54)
    ZSEAnalyze_(args, 16, 216)
    # ZSEAnalyze_(args, 23, 9)
    # ZSEAnalyze_(args, 23, 216)
    # print(args.net)
