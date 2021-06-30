
import os
from matplotlib import pyplot as plt
import torch
import numpy as np

from utils.logger import Log
from conf import FLAGS, COMP_TYPE

CONF_SZ = 3

def SaveStackedGraph(xlabels, data, mode="percentage", title="", save=""):
    if mode == "percentage":
        percent = data / data.sum(axis=0).astype(float) * 100
    else:
        percent = data
    
    # Set figure
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    x = np.arange(data.shape[1])
    # Set the colors
    colors = ['#f00']
    for i in range(data.shape[0]):
        colors.append("{}".format(0.5 - 0.5 * float(i) / float(data.shape[0])))
    ax.stackplot(x, percent, colors=colors)
    ax.set_title(title)
    # Set labels
    if mode == "percentage":
        ax.set_ylabel('Percent (%)')
    else:
        ax.set_ylabel('Count')
    plt.xlabel("", labelpad=30)
    # plt.tight_layout(pad=6.0)

    # Set X labels
    plt.xticks(x,xlabels, rotation=45)
    fig.autofmt_xdate()
    ax.margins(0, 0) # Set margins to avoid "whitespace"
    
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/"+save + ".png")




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

            if args.zse_print_mode == "format":
                for i in d:
                    Log.Print("{}".format(i),end="\t")
                Log.Print("")
            if args.zse_print_mode == "all":
                Log.Print("[{:7.3f}%]{:8d}/{:8d}, {}".format(d[-1]/(d.sum())*100,d[-1],d.sum(),d))
    # Save figures
    if args.zse_graph_mode != "none":
        SaveStackedGraph(args.layer_list_short, np.flip(stat_data.transpose(),axis=0),
                mode=args.zse_graph_mode,
                title="{}, Bit={}, Group={}".format(args.model, bits, g_size),
                save="{}_{}_{}".format(args.model, bits, g_size))
    if args.zse_print_mode in ["sum", "format", "all"]:
        Log.Print("[{:7.3f}%]{:10d}/{:10d}, {}".format(res[-1]/(res.sum())*100,res[-1],res.sum(),res))            
        Log.Print("")


def GetZSE(inp_n, group_mantissa, group_size, group_direction):
    # Save original shape
    inp_shape = inp_n.shape
    # STEP 1 : Pre-process array to match with group size
    # Pad Array to do the grouping correctly
    g_kernel = int(group_size / CONF_SZ / CONF_SZ)
    if inp_n.shape[0] % g_kernel != 0:
        inp_n = np.pad(inp_n, ((0,g_kernel-inp_n.shape[0]%g_kernel),(0,0),(0,0),(0,0)))
    if inp_n.shape[1] % g_kernel != 0:
        inp_n = np.pad(inp_n, ((0,0),(0,g_kernel-inp_n.shape[1]%g_kernel),(0,0),(0,0)))
    if inp_n.shape[2] % CONF_SZ != 0 or inp_n.shape[CONF_SZ] % CONF_SZ != 0:
        inp_n = np.pad(inp_n, ((0,0),(0,0),(0,CONF_SZ-inp_n.shape[2]%CONF_SZ),(0,CONF_SZ-inp_n.shape[3]%CONF_SZ)))
    inp_p_shape = inp_n.shape

    # transposing array to desired grouping direction
    # TODO : Prevent overflow caused by mapping next kernel map on FX, FY, FC mode
    # FX, FY, FC Mode have to reshape array to dimension of 8 to manipulate more easily
    #   Code modified from https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440
    if group_direction == 0: # WI Mode, If kernel size is not 3, it will not work properly
        inp_n = np.transpose(inp_n, (2,3,0,1))
    elif group_direction == 1: # WO Mode, If kernel size is not 3, it will not work properly
        inp_n = np.transpose(inp_n, (2,3,1,0))
    elif group_direction == 10: # FX Mode
        inp_n = inp_n.reshape((inp_n.shape[0], 1, inp_n.shape[1], 1, inp_n.shape[2]//CONF_SZ, CONF_SZ, inp_n.shape[3]//CONF_SZ, CONF_SZ))
        inp_n = inp_n.transpose((0,2,4,6,5,7,1,3))
    elif group_direction == 11: # FY Mode
        inp_n = inp_n.reshape((inp_n.shape[0], 1, inp_n.shape[1], 1, inp_n.shape[2]/CONF_SZ, CONF_SZ, inp_n.shape[3]//CONF_SZ, CONF_SZ))
        inp_n = inp_n.transpose((0,2,6,4,5,7,1,3))
    elif group_direction == 12:
        inp_n = inp_n.reshape((inp_n.shape[0], 1, inp_n.shape[1], 1, inp_n.shape[2]//CONF_SZ, CONF_SZ, inp_n.shape[3]//CONF_SZ, CONF_SZ))
        inp_n = inp_n.transpose((0,4,6,2,5,7,1,3))
    else:
        raise ValueError("group_direction not supported")
    # Save modified shape
    inp_m_shape = inp_n.shape
    # Flatten
    inp_n = np.reshape(inp_n, (np.product(inp_n.shape),))
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32) 

    e_mask = np.full(v.shape, 0x7f800000, dtype=np.uint32)
    e_ = np.bitwise_and(v, e_mask)
    # Get the max value
    # IDEA : send shift code to back, maybe that's faster
    np.right_shift(e_, 23, out=e_)
    e_ = e_.astype(np.int32)

    e_mask = np.full(v.shape, 0xff, dtype=np.uint32)
    e_ = np.bitwise_and(e_, e_mask)

    # Reshape array to group size to get max values
    m_ = np.reshape(e_, (-1, group_size))
    # get the max value of each blocks
    m_ = np.amax(m_, axis=1)
    m_ = np.reshape(m_, (1,-1))
    # Revert back to original size
    m_ = np.repeat(m_, group_size, axis=1)
    # Match shape back to input
    # Difference of the exponent, -1 is applied because for more accurate hardware-wise simulation
    # On hardware, mantissa bits have to store the 1 from the IEEE Standard
    e_ = group_mantissa - (m_ - e_) - 1    

    # Reshape back here
    r = e_.reshape(inp_m_shape)

    # Transpose and reshape back to original array
    if group_direction == 0: # WI
        r = np.transpose(r, (2,3,0,1))
    elif group_direction == 1: # WO
        r = np.transpose(r, (3,2,0,1))
    elif group_direction == 10: # FX
        r = r.transpose((0,6,1,7,2,4,3,5))
        r = r.reshape(inp_p_shape)
    elif group_direction == 11: # FY Mode
        r = r.transpose((0,6,1,7,3,4,2,5))
        r = r.reshape(inp_p_shape)
    elif group_direction == 12:
        r = r.transpose((0,6,3,7,1,4,2,5))
        r = r.reshape(inp_p_shape)
    # Revert padding
    if inp_p_shape != inp_shape:
        r = r[:inp_shape[0],:inp_shape[1],:inp_shape[2],:inp_shape[3]]

    # Get the stats
    r = r.flatten()
    res = np.zeros(group_mantissa+1, dtype=np.int32)
    for i in range(group_mantissa+1):
        res[i] = (r == i).sum()
    res[-1] = (r < 0).sum()
    # print(total, res, res.sum())
    # print("bit={}, sz={}, dim={} = {} / {}".format(group_mantissa, group_size, inp_shape, np.asarray(inp_shape).prod(), res.sum()))
    return res

from functions import DictKey


class ZSEObject_:
    def __init__(self) -> None:
        self.data = dict()
        self.data_fi = None

    # AddData will add data to the zse object
    def AddData(self, inp, group_mantissa, group_size, group_direction, type):
        # print(inp.shape)
        typename = DictKey(COMP_TYPE, type)
        if typename in ["fw", "fi", "bio"]:
            res = GetZSE(inp, group_mantissa, group_size, group_direction)
            if typename not in self.data:
                self.data[typename] = np.zeros(group_mantissa+1)
            self.data[typename] += res
    
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        s = "\nZSE Analyze Result"
        for key, value in self.data.items():
            total = value.sum()
            error = value[-1]
            s += "\n%s: %16d / %16d (%2.5f)\n%d\t"%(key, error, total, float(error/total*100), value[-1])
            for i in value[:-1]:
                s += "%d\t"%i
        s +="\n"
        return s

        
ZSEObject = ZSEObject_()

def ZSEAnalyze(args):
    Log.SetPrintCurrentTime(False)
    Log.SetPrintElapsedTime(False)
    FLAGS.ZSE = True

    args.net.load_state_dict(torch.load(args.save_file))
    args.net.eval()

    count = 10
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

        Log.Print("%d/%d Forward Processed"%(i+1, count))
        if i == count - 1:
            break

    Log.Print(str(ZSEObject))

    # parameters = torch.load(args.save_file).items()
    
    Log.Print("")
    # for bits in [4]:
    #     for g_size in [36]:
    # for bits in [4, 5, 6, 7, 8]:
    #     for g_size in [36, 54, 72]:
    # ZSEAnalyze_(args, 4, 36)
    # ZSEAnalyze_(args, 4, 144)
    # ZSEAnalyze_(args, 8, 54)
    # ZSEAnalyze_(args, 8, 216)
    # ZSEAnalyze_(args, 16, 54)
    # ZSEAnalyze_(args, 16, 216)
    
    # ZSEAnalyze_(args, 23, 9)
    # ZSEAnalyze_(args, 23, 216)
    # print(args.net)
