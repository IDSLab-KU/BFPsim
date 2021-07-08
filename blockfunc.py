import torch
import numpy as np
from conf import FLAGS
from utils.ZSEAnalyze import ZSEObject
import ctypes

fp32_mask = [0,
    0x00400000, 0x00600000, 0x00700000, 0x00780000,
    0x007c0000, 0x007e0000, 0x007f0000, 0x007f8000,
    0x007fc000, 0x007fe000, 0x007ff000, 0x007ff800,
    0x007ffc00, 0x007ffe00, 0x007fff00, 0x007fff80,
    0x007fffc0, 0x007fffe0, 0x007ffff0, 0x007ffff8, 0x007fffff]

fp64_mask = [0,
    0x0040000000000000, 0x0060000000000000, 0x0070000000000000, 0x0078000000000000,
    0x007c000000000000, 0x007e000000000000, 0x007f000000000000, 0x007f800000000000,
    0x007fc00000000000, 0x007fe00000000000, 0x007ff00000000000, 0x007ff80000000000,
    0x007ffc0000000000, 0x007ffe0000000000, 0x007fff0000000000, 0x007fff8000000000]

from numba import jit, cuda

@cuda.jit
def make_groups_4d_internal(v, dim, bs, gs, group_mantissa):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x 
    
    idx0o = (idx // (bs[3] * bs[2] * bs[1])) % bs[0] * gs[0]
    idx1o = (idx // (bs[3] * bs[2])) % bs[1] * gs[1]
    idx2o = (idx // bs[3]) % bs[2] * gs[2]
    idx3o = idx % bs[3] * gs[3]

    M = 0
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim[0]:
            break
        for idx1 in range(idx1o, idx1o + gs[1]):
            if idx1 >= dim[1]:
                break
            for idx2 in range(idx2o, idx2o + gs[2]):
                if idx2 >= dim[2]:
                    break
                for idx3 in range(idx3o, idx3o + gs[3]):
                    if idx3 >= dim[3]:
                        break
                    e = (v[idx0,idx1,idx2,idx3] >> 23 ) & 0xff
                    if M < e:
                        M = e

    # Replace that area
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim[0]:
            break
        for idx1 in range(idx1o, idx1o + gs[1]):
            if idx1 >= dim[1]:
                break
            for idx2 in range(idx2o, idx2o + gs[2]):
                if idx2 >= dim[2]:
                    break
                for idx3 in range(idx3o, idx3o + gs[3]):
                    if idx3 >= dim[3]:
                        break
                    e = (v[idx0,idx1,idx2,idx3] >> 23 ) & 0xff
                    if M - e <= group_mantissa - 1:
                        v[idx0,idx1,idx2,idx3] = v[idx0,idx1,idx2,idx3] & (0xffffffff << (24 - group_mantissa + M - e))
                    else:
                        v[idx0,idx1,idx2,idx3] = 0

arrDict = dict()

# set_mantissa_tensor : set to tensor or numpy array to speicific mantissa bits 
# TODO : Set direction of grouping
def set_mantissa_tensor(inp, group_mantissa):
    if inp.is_cuda:
        inp_n = inp.cpu().numpy() # inp_n = inp # For debug,
    else:
        inp_n = inp.numpy()
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32) 
    # Generate mask
    r_mask = np.asarray(np.full(v.shape, 0x007fffff, dtype=np.uint32))
    # Shift to make reversed mask
    r_mask = np.right_shift(r_mask, group_mantissa)
    # Get the reversed mask
    r_mask = np.invert(r_mask)
    # And operation to remove mantissa
    r_ = np.bitwise_and(v, r_mask)
    # revert to original np.float32 
    r = np.frombuffer(r_, dtype=np.float32)
    if inp.is_cuda:
        return torch.from_numpy(r.reshape(inp_n.shape)).cuda()
    else:
        return torch.from_numpy(r.reshape(inp_n.shape))

threadsperblock = 1024

# Grouping tensor for fully connected layer. Direction is not important
def make_groups_tensor_fc(inp, group_mantissa, group_size, group_direction):
    if group_size == 1:
        return set_mantissa_tensor(inp, group_mantissa)
    # Convert tensor to numpy array
    if inp.is_cuda:
        inp_n = inp.cpu().numpy()
    else:
        inp_n = inp.numpy()

    inp_m_shape = inp_n.shape
    # Flatten
    inp_n = np.reshape(inp_n, (np.product(inp_n.shape),))
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32) 

    gs = (1, group_size//9, 3, 3)
    
    # STEP 2 : gpu computation
    r_ = cuda.to_device(v)
    blockspergrid = (v.size + (threadsperblock - 1)) // threadsperblock
    make_groups_3d_internal[blockspergrid, threadsperblock](r_, group_size, gs, group_mantissa)
    r__ = r_.copy_to_host()

    # STEP 3 : reverting array
    # revert to original np.float32 
    r = np.frombuffer(r__, dtype=np.float32)
    # revert back to original shape
    r = r.reshape(inp_m_shape)

    if inp.is_cuda:
        return torch.from_numpy(r).cuda()
    else:
        return torch.from_numpy(r)

import numba

# make_group_tensor : Group values as same exponent bits, which shifts mantissa
def make_groups_tensor(inp, group_mantissa, group_size, group_direction, type = -1):
    if group_size == 1:
        return set_mantissa_tensor(inp, group_mantissa)
    
 
    blockspergrid = (inp.size()[0]*inp.size()[1]*inp.size()[2]*inp.size()[3] +  (threadsperblock - 1)) // threadsperblock
    inp = inp.view(inp.size(), torch.int32)
    if group_direction == 0: # WI Mode, If kernel size is not 3, it will not work properly
        gs = (group_size//9, 1, 3, 3)
    elif group_direction == 1: # WO Mode, If kernel size is not 3, it will not work properly
        gs = (1, group_size//9, 3, 3)
    elif group_direction == 10: # FX Mode
        gs = (1, 1, group_size//3, 3) # Group's size may small if image size is smaller than group_size//3
    elif group_direction == 11: # FY Mode
        gs = (1, 1, 3, group_size//3) # Group's size may small if image size is smaller than group_size//3
    elif group_direction == 12: # FC Mode
        gs = (1, group_size//9, 3, 3)
    else:
        raise ValueError("group_direction not supported")
    
    bs = ((inp.size()[0]-1)//gs[0]+1, (inp.size()[1]-1)//gs[1]+1, (inp.size()[2]-1)//gs[2]+1, (inp.size()[3]-1)//gs[3]+1)

    # inp = cuda.to_device(inp)
    make_groups_4d_internal[blockspergrid, threadsperblock](inp, inp.size(), bs, gs, group_mantissa)
    # inp = inp.copy_to_host()

    # inp = numba.carray(inp, inp_n.shape, dtype=torch.float32)
    inp.view(dtype=torch.float32)
    return inp
