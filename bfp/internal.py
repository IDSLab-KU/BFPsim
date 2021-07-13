import torch
import numpy as np
import ctypes

from conf import FLAGS, CUDA_THREADSPERBLOCK


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
import numba

@cuda.jit
# TODO : make another function to just grouping tensor...?
def make_groups_3d_internal(v, dim, bs, gs, group_mantissa):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x 
    
    idx0o = (idx // (bs[2] * bs[1])) % bs[0] * gs[0]
    idx1o = (idx // (bs[2])) % bs[1] * gs[1]
    idx2o = idx % bs[2] * gs[2]

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
                e = (v[idx0,idx1,idx2] & 0x7f800000 ) >> 23
                if e < 0:
                    e = -e
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
                e = (v[idx0,idx1,idx2] & 0x7f800000 ) >> 23
                k = group_mantissa - M + e - 1
                if 0 <= k:
                    v[idx0,idx1,idx2] = v[idx0,idx1,idx2] & (0xffffffff << (23 - k))
                else:
                    v[idx0,idx1,idx2] = 0

@cuda.jit
# TODO : make another function to just grouping tensor...?
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
                    e = (v[idx0,idx1,idx2,idx3] & 0x7f800000 ) >> 23
                    if e < 0:
                        e = -e
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
                    e = (v[idx0,idx1,idx2,idx3] & 0x7f800000 ) >> 23
                    k = group_mantissa - M + e - 1
                    if 0 <= k:
                        v[idx0,idx1,idx2,idx3] = v[idx0,idx1,idx2,idx3] & (0xffffffff << (23 - k))
                    else:
                        v[idx0,idx1,idx2,idx3] = 0



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

# make_group_tensor : Group values as same exponent bits, which shifts mantissa
def make_groups_tensor(inp, group_mantissa, group_dim, type = -1):
    # Convert tensor to numpy array
    if inp.is_cuda:
        inp_n = inp.cpu().numpy()
    else:
        inp_n = inp.numpy()
    # Save original shape
    inp_shape = inp_n.shape
    # STEP 1 : Pre-process array to match with group size
    # Pad Array to do the grouping correctly
    g_kernel = int(max(group_dim[0], group_dim[1]) / 3 / 3)
    if g_kernel != 0 and inp_n.shape[0] % g_kernel != 0:
        inp_n = np.pad(inp_n, ((0,g_kernel-inp_n.shape[0]%g_kernel),(0,0),(0,0),(0,0)))
    if g_kernel != 0 and inp_n.shape[1] % g_kernel != 0:
        inp_n = np.pad(inp_n, ((0,0),(0,g_kernel-inp_n.shape[1]%g_kernel),(0,0),(0,0)))
    if inp_n.shape[2] % 3 != 0 or inp_n.shape[3] % 3 != 0:
        inp_n = np.pad(inp_n, ((0,0),(0,0),(0,3-inp_n.shape[2]%3),(0,3-inp_n.shape[3]%3)))
    inp_p_shape = inp_n.shape

    # transposing array to desired grouping direction
    # TODO : Prevent overflow caused by mapping next kernel map on FX, FY, FC mode
    # FX, FY, FC Mode have to reshape array to dimension of 8 to manipulate more easily
    #   Code modified from https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440
    if group_dim[1] == 1: # WI Mode, If kernel size is not 3, it will not work properly
        inp_n = np.transpose(inp_n, (2,3,0,1))
    elif group_dim[0] == 1 and group_dim[1] == 1: # FX Mode
        inp_n = inp_n.reshape((inp_n.shape[0], 1, inp_n.shape[1], 1, inp_n.shape[2]//3, 3, inp_n.shape[3]//3, 3))
        inp_n = inp_n.transpose((0,2,4,6,5,7,1,3))
    else:
        inp_n = inp_n.reshape((inp_n.shape[0], 1, inp_n.shape[1], 1, inp_n.shape[2]//3, 3, inp_n.shape[3]//3, 3))
        inp_n = inp_n.transpose((0,4,6,2,5,7,1,3))
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
    blockspergrid = (v.size + (CUDA_THREADSPERBLOCK - 1)) // CUDA_THREADSPERBLOCK
    make_groups_gpu[blockspergrid, CUDA_THREADSPERBLOCK](r_, group_mantissa, group_dim[0]*group_dim[1]*group_dim[2]*group_dim[3])
    r__ = r_.copy_to_host()
    
    # Previous STEP 2 : make groups and adjust mantissa
    # r__ = _make_groups(v, group_mantissa, group_size)

    # STEP 3 : reverting array
    # revert to original np.float32 
    r = np.frombuffer(r__, dtype=np.float32)
    # revert back to original shape
    r = r.reshape(inp_m_shape)
    # Transpose and reshape back to original array
    if group_dim[1] == 1: # WI
        r = np.transpose(r, (2,3,0,1))
    elif group_dim[0] == 1 and group_dim[1] == 1: # FX
        r = r.transpose((0,6,1,7,2,4,3,5))
        r = r.reshape(inp_p_shape)
    else:
        r = r.transpose((0,6,3,7,1,4,2,5))
        r = r.reshape(inp_p_shape)
    # Revert padding
    if inp_p_shape != inp_shape:
        r = r[:inp_shape[0],:inp_shape[1],:inp_shape[2],:inp_shape[3]]

    if inp.is_cuda:
        return torch.from_numpy(r).cuda()
    else:
        return torch.from_numpy(r)