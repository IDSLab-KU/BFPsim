

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
