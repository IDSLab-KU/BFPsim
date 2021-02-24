

mantissa_mask = [     0x0
                ,     0x1,     0x3,     0x7,     0xf,    0x1f,    0x3f,    0x7f,    0xff
                ,   0x1ff,   0x3ff,   0x7ff,   0xfff,  0x1fff,  0x3fff,  0x7fff,  0xffff
                , 0x1ffff, 0x3ffff, 0x7ffff, 0xfffff,0x1fffff,0x3fffff,0x7fffff]

import numpy as np

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

# set_mantissa
# set to float np array to speicific mantissa bits 
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

# group
# Group values as same exponent bits, which shifts mantissa
# TODO : Set direction of grouping
def make_groups(inp, mb, out = None, group_size = 36, direction = None, silent=True):
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