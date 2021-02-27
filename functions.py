

mantissa_mask = [     0x0
                ,     0x1,     0x3,     0x7,     0xf,    0x1f,    0x3f,    0x7f,    0xff
                ,   0x1ff,   0x3ff,   0x7ff,   0xfff,  0x1fff,  0x3fff,  0x7fff,  0xffff
                , 0x1ffff, 0x3ffff, 0x7ffff, 0xfffff,0x1fffff,0x3fffff,0x7fffff]

import numpy as np
import torch

import struct

def FloatToBits(f):
    s = struct.pack('>f', f)
    if f < 0:
        return - struct.unpack('>l', s)[0] ^ 0xffffffff
    else:
        return struct.unpack('>l', s)[0]
def BitsToFloat(f):
    # print("{:20d}".format(f), format(f,"032b"))
    if f>>31 == 1:
        f = - (f ^ 0xffffffff)
    # TODO : make this work
    s = struct.pack('>l', f)
    return struct.unpack('>f', s)[0]

# set_mantissa : set to float np array to speicific mantissa bits 
def set_mantissa(inp, out = None, mb = 8):
    if type(out) != np.ndarray:
        outx = np.zeros(inp.shape,dtype=inp.dtype)
    else:
        outx = inp

    for idx, val in np.ndenumerate(inp):
        v = FloatToBits(val)
        # only bring mantissa
        m = ((v & 0x007fffff) >> (23 - mb)) << (23 - mb)
        v = v & 0xff800000 | m
        outx[idx] = BitsToFloat(v)
    
    if type(out) != np.ndarray:
        return outx

# make_groups : Group values as same exponent bits, which shifts mantissa
# TODO : Set direction of grouping
def make_groups(inp, mb, out = None, group_size = 36, direction = None):
    if type(out) != np.ndarray:
        outx = np.zeros(inp.shape,dtype=inp.dtype)
    else:
        outx = inp
    i_ = [None] * group_size
    s_ = [None] * group_size
    e_ = [None] * group_size
    m_ = [None] * group_size
    
    vdx = 0
    for idx, val in np.ndenumerate(inp):
        v = FloatToBits(val)
        i_[vdx] = idx
        s_[vdx] = v >> 31
        e_[vdx] = (v & 0x7f800000) >> 23
        m_[vdx] = (v & 0x007fffff) >> (23 - mb)
        vdx += 1
        # Find the max size
        if vdx == group_size:
            vdx = 0
            me = np.amax(e_)
            for iidx in range(group_size):
                m_[iidx] = (m_[iidx] >> (me - e_[iidx])) << (me - e_[iidx])
                v = (s_[iidx] << 31) | (e_[iidx] << 23) | (m_[iidx] << (23 - mb))
                outx[i_[iidx]] = BitsToFloat(v)
                # print("{} {:8.4f} > {:8.4f}".format(me-e_[iidx],inp[i_[iidx]], outx[i_[iidx]]), end="\t")
            """
            if not silent:
                print("->",end="")
                for i in range(len(m_)):
                    print("{}_{:012b}".format(me - e_[i],m_[i]),end=" ")
                print()
            """
    
    # handling last leftover values
    if vdx != 0:
        me = np.amax(e_)
        for iidx in range(group_size):
            m_[iidx] = (m_[iidx] >> (me - e_[iidx])) << (me - e_[iidx])
            v = (s_[iidx] << 31) | (e_[iidx] << 23) | (m_[iidx] << (23 - mb))
            outx[i_[iidx]] = BitsToFloat(v)
    if type(out) != np.ndarray:
        return outx




def p(v):
    for i in v:
        print(format(i,"064b"), end="\n")
    print()

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
def set_mantissa_tensor(inp, mb):
    inp_n = inp.numpy() # inp_n = inp # For debug,
    # Convert to byte stream
    st = inp_n.tobytes() 
    # Set to uint32 array to easy computing
    v = np.frombuffer(st, dtype=np.uint32) 
    # And operation to remove mantissa
    r_mask = np.full(v.shape, 0xff800000 | fp32_mask[mb], dtype=np.uint32)
    r_ = np.bitwise_and(v, r_mask)
    # revert to original np.float32 
    r = np.frombuffer(r_, dtype=np.float32)
    return torch.from_numpy(r.reshape(inp_n.shape))

# make_group_tensor : Group values as same exponent bits, which shifts mantissa
# TODO : Set direction of grouping
def make_groups_tensor(inp, mb, group_size = 36, direction = None):
    inp_n = inp.numpy() # inp_n = inp # For debug,
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
    # Match shape to divisible to group size
    m_ = np.append(e_, np.zeros(group_size - e_.shape[0] % group_size, dtype=np.uint32))
    m_ = np.reshape(m_, (group_size, -1))
    # get the max value of each blocks
    m_ = np.amax(m_, axis=0)
    # Revert back to original size
    m_ = np.repeat(m_, group_size)
    # Match shape back to input
    m_ = m_[:e_.shape[0]]

    # Difference of the exponent
    # -1 is for on grouping, IEEE's basic mantissa bit has to be included to the value, so...
    e_ = mb - 1 - (m_ - e_)
    # Clip the negative value (I know this is not smarter way)
    e_[e_ > 0xff] = 0
    # np.clip(e_, 0, 0xff, out=e_) # Options...
    r_mask = np.full(v.shape, 0x007fffff, dtype=np.uint32)
    # Shift to make reversed mask
    np.right_shift(r_mask, e_, out=r_mask)
    # Get the reversed mask
    np.invert(r_mask, out=r_mask)
    r_ = np.bitwise_and(v, r_mask)
    # revert to original np.float32 
    r = np.frombuffer(r_, dtype=np.float32)
    return torch.from_numpy(r.reshape(inp_n.shape))