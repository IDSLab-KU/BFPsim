import torch
import numpy as np
from conf import FLAGS
from utils.ZSEAnalyze import ZSEObject

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


# Basic size dimension, most of code will not work if this value is changed
# This is set as constant because of expandability
CONF_SZ = 3

from numba import jit, cuda
# jit function to improve speed
@jit(nopython=True)
def _make_groups(v, group_mantissa, group_size):
    r_ = v.copy()
    for i in range(v.shape[0] // group_size):
        M = 0
        for ii in range(group_size):
            if M < (v[i*group_size+ii] >> 23) & 0xff:
                M = (v[i*group_size+ii] >> 23) & 0xff
        for ii in range(group_size):
            if (M - (v[i*group_size+ii] >> 23) & 0xff) <= group_mantissa - 1:
                r_[i*group_size+ii] = v[i*group_size+ii] & (0xffffffff << (23 - group_mantissa + 1 + (M - (v[i*group_size+ii] >> 23) & 0xff)))
            else:
                r_[i*group_size+ii] = 0x00000000
    return r_


@jit(nopython=True)
def _Internal_Group_WI(v, group_mantissa, group_size):
    r_ = v.copy()
    

@cuda.jit
def make_groups_gpu(arr, group_mantissa, group_size):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Max value of one block
    gi = tx + ty * bw
    if gi < arr.size // group_size:
        M = 0
        # Get max mantissa
        for ii in range(group_size):
            e = (arr[gi*group_size+ii] >> 23) & 0xff
            if M < e:
                M = e
        # Assign float values
        for ii in range(group_size):
            arri = gi*group_size+ii
            e = (arr[arri] >> 23) & 0xff
            if M - e <= group_mantissa - 1:
                arr[arri] = arr[arri] & (0xffffffff << (24 - group_mantissa + M - e))
                pass
            else:
                arr[arri] = 0

threadsperblock = 128

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

    # STEP 2 : gpu computation
    r_ = cuda.to_device(v)
    blockspergrid = (v.size + (threadsperblock - 1)) // threadsperblock
    make_groups_gpu[blockspergrid, threadsperblock](r_, group_mantissa, group_size)
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

# make_group_tensor : Group values as same exponent bits, which shifts mantissa
def make_groups_tensor(inp, group_mantissa, group_size, group_direction, type = -1):
    # return set mantissa if group size is 1
    if group_size == 1:
        return set_mantissa_tensor(inp, group_mantissa)
    # Convert tensor to numpy array
    if inp.is_cuda:
        inp_n = inp.cpu().numpy()
    else:
        inp_n = inp.numpy()
    # ZSE Handling
    if FLAGS.ZSE:
        ZSEObject.AddData(inp_n, group_mantissa, group_size, group_direction, type)
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

    # STEP 2 : gpu computation
    r_ = cuda.to_device(v)
    blockspergrid = (v.size + (threadsperblock - 1)) // threadsperblock
    make_groups_gpu[blockspergrid, threadsperblock](r_, group_mantissa, group_size)
    r__ = r_.copy_to_host()
    
    # Previous STEP 2 : make groups and adjust mantissa
    # r__ = _make_groups(v, group_mantissa, group_size)

    # STEP 3 : reverting array
    # revert to original np.float32 
    r = np.frombuffer(r__, dtype=np.float32)
    # revert back to original shape
    r = r.reshape(inp_m_shape)
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

    if inp.is_cuda:
        return torch.from_numpy(r).cuda()
    else:
        return torch.from_numpy(r)
