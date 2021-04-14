# utils.py

import torch

from log import Log

from blockfunc import GetZeroSettingError
import numpy as np
from functions import SaveStackedGraph



def SaveData(args):
    pass



def ZeroTest(args):

    Log.SetPrintCurrentTime(False)
    Log.SetPrintElapsedTime(False)

    parameters = torch.load(args.save_file).items()
    
    layer_count = 0
    layer_list = []
    layer_list_short = []
    Log.Print("List of weight data's name")
    for name, param in parameters:
        n = name.split(".")
        # Normal model name convention
        if args.model == "ResNet18" and args.zt_bf == False:
            condition = "layer" in n[1] and "conv" in n[3] and "weight" in n[4]
            if condition:
                short_name = n[1]+"_"+n[2]+"_"+n[3]
        elif args.model == "ResNet18" and args.zt_bf == True:
            condition = "layer" in n[0] and "conv" in n[2] and "weight" in n[3]
            if condition:
                short_name = n[0]+"_"+n[1]+"_"+n[2]
        else:
            raise NotImplementedError("model and layer conf not implemented")

        if condition:
            Log.Print("{}({})".format(name, param.size()))
            layer_list.append(name)
            layer_list_short.append(short_name)
            layer_count += 1
        
    Log.Print("")

    # for bits in [4]:
    #     for g_size in [36]:
    for bits in [4, 5, 6, 7, 8]:
        for g_size in [36, 54, 72]:
            stat_data = np.zeros((layer_count, bits+1),dtype=np.int32)
            Log.Print("Mantissa bits {}, Group size {}".format(bits, g_size))
            res = np.zeros(bits+1, dtype=np.int32)
            ind = 0
            for name, param in parameters:
                if name in layer_list:
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
                SaveStackedGraph(layer_list_short, np.flip(stat_data.transpose(),axis=0),
                        mode=args.zt_graph_mode,
                        title="{}, Bit={}, Group={}".format(args.model, bits, g_size),
                        save="{}_{}_{}".format(args.model, bits, g_size))
            if args.zt_print_mode in ["sum", "format", "all"]:
                Log.Print("[{:7.3f}%]{:10d}/{:10d}, {}".format(res[-1]/(res.sum())*100,res[-1],res.sum(),res))            
                Log.Print("")
    # print(args.net)
