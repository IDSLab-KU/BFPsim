import torch
import numpy as np
from functions import DIR_DICT

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
    inp_n = inp.numpy() # inp_n = inp # For debug,
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
    return torch.from_numpy(r.reshape(inp_n.shape))

# Basic size dimension, most of code will not work if this value is changed
# This is set as constant because of expandability
CONF_SZ = 3


# _make_group_tensor : Group values as same exponent bits, which shifts mantissa
def make_groups_tensor(inp, group_mantissa, group_size, group_direction):
    # Convert tensor to numpy array
    inp_n = inp.numpy()
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
        inp_n = inp_n.reshape((inp_n.shape[0], 1, inp_n.shape[1], 1, inp_n.shape[2]/CONF_SZ3, CONF_SZ, inp_n.shape[3]//CONF_SZ, CONF_SZ))
        inp_n = inp_n.transpose((0,2,6,4,5,7,1,3))
    elif group_direction == 12:
        inp_n = inp_n.reshape((inp_n.shape[0], 1, inp_n.shape[1], 1, inp_n.shape[2]//CONF_SZ, CONF_SZ, inp_n.shape[3]//CONF_SZ, CONF_SZ))
        inp_n = inp_n.transpose((0,4,6,2,5,7,1,3))
    else:
        raise ValueError("group_direction not supported")
    # Save modified shape
    inp_m_shape = inp_n.shape

    # STEP 2 : make groups and adjust mantissa
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32) 
    # Extract exponent
    e_mask = np.full(v.shape, 0x7f800000, dtype=np.uint32)
    e_ = np.bitwise_and(v, e_mask)
    # Get the max value
    # IDEA : send shift code to back, maybe that's faster
    np.right_shift(e_, 23, out=e_)
    # Reshape arrat to group size to get max values
    m_ = np.reshape(e_, (group_size, -1))
    # get the max value of each blocks
    m_ = np.amax(m_, axis=0)
    # Revert back to original size
    m_ = np.repeat(m_, group_size)
    # Match shape back to input
    m_ = m_[:e_.shape[0]]
    # Difference of the exponent, -1 is applied because for more accurate hardware-wise simulation
    # On hardware, mantissa bits have to store the 1 from the IEEE Standard
    e_ = group_mantissa - (m_ - e_) - 1
    r_mask = np.full(v.shape, 0x007fffff, dtype=np.uint32)
    # When mantissa have to shift more than precision bits, total value is zero on hardware.
    r_mask[e_ > 0xff] = 0xffffffff
    # Clip the negative value, Maybe more smarter way...? -> np.clip(e_, 0, 0xff, out=e_)
    e_[e_ > 0xff] = 0
    # Shift to make reversed mask
    np.right_shift(r_mask, e_, out=r_mask)
    # Get the reversed mask
    np.invert(r_mask, out=r_mask)
    # appling mask to data
    r_ = np.bitwise_and(v, r_mask)

    # STEP 3 : reverting array
    # revert to original np.float32 
    r = np.frombuffer(r_, dtype=np.float32)
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

    return torch.from_numpy(r)
