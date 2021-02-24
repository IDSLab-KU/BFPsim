"""
bfarray.py
Brain floating point array supporting grouping, computing between them
Mantissa bit can be set from 1 to 23, which smaller mantissa bits can lead incorrect values.
BFloat supports:
 - Assign value from float, int, and BFloat
 - Basic arithmatic expression: add, sub, mul, div(truediv)
 - Basic bitwise operation: and, or, xor
 - type casting to int and float. Converting to int will set bit value of float.
 - Comparsion operators: lt, gt, le, ge, eq, ne
 - Unary operators: neg, pos, invert, abs
 - If Bfloat is printed, it will print the bit value it holds.
   Print float(BFloat) to print float value
"""
import struct

def floatToBits(f):
    s = struct.pack('>f', f)
    if f < 0:
        return - struct.unpack('>l', s)[0] ^ 0xffffffff
    else:
        return struct.unpack('>l', s)[0]

def BitToFloats(f):
    # print("{:20d}".format(f), format(f,"032b"))
    if f>>31 == 1:
        f = - (f ^ 0xffffffff)
    # TODO : make this work
    s = struct.pack('>l', f)
    return struct.unpack('>f', s)[0]

mantissa_mask = [     0x0
                ,     0x1,     0x3,     0x7,     0xf,    0x1f,    0x3f,    0x7f,    0xff
                ,   0x1ff,   0x3ff,   0x7ff,   0xfff,  0x1fff,  0x3fff,  0x7fff,  0xffff
                , 0x1ffff, 0x3ffff, 0x7ffff, 0xfffff,0x1fffff,0x3fffff,0x7fffff]

import numpy as np



from bfloat import BFloat
"""
    # Representors
    def __repr__(self):
        return str(self)
    def __str__(self):
        # Print virtual exponent and virtual mantissa
        e = self.e + self.vs
        m = (self.m | (0x1 << self.mb)) >> (1 + self.vs)
        s = '-' if self.s else '+'
        return '{}{}_{}'.format(s, format(e,"02x"), format(m,"0{}x".format((self.mb-1)//4+1)))
"""

class BFArray(object):
    def __init__(self, size, mb=8, group_size=36):
        '''
        size: Size of array provided with tuple
        mb: Size of mantissa bits
        '''
        # Size of the array
        # On linear array, it's (output, input)
        self.size = size
        self.shape = self.size # For easy usage
        self.count = 1
        for i in self.size:
            self.count *= i
        assert mb < 24 and mb > 0, "Mantissa bit not supported"
        self.mb = mb

        self.group_size = group_size # 36

        # Store exponent, mantissa
        # Sign is stored on mantissa
        self.e = np.zeros(size,dtype=np.uintc)
        self.m = np.zeros(size,dtype=np.intc)

    def __str__(self):
        return 'BFArray Object size of {}'.format(self.size)

    # Initialize with grouping numbers
    def initialize(self, method=0):
        # TODO : Implement array with more than size of 2
        self.e = (np.random.randn(self.size[0], self.size[1]) + 128).astype(int)
        self.m = np.random.randint(0,255,self.size,dtype=np.intc) - 128
        # self.m = np.random.randint(0,127,self.size,dtype=np.intc)
        self.make_groups()

    def make_groups(self):
        # Group weights as same exponent bits, which shifts mantissa
        # print(self.s.shape)
        e_ = np.reshape(self.e, (self.count))
        m_ = np.reshape(self.m, (self.count))

        mi, me = 0, 0
        for i in range(self.count//self.group_size+1):
            istart = i*self.group_size
            iend = min(self.count, istart+self.group_size)
            # Stage 1: find maximum exponent on group
            me = np.amax(e_[istart:iend])
            # Set the exponent and shift mantissa
            for ind in range(istart, iend):
                e_[ind] = me
                m_[ind] = m_[ind] >> (me - e_[ind])
        self.e = np.reshape(e_, self.size)
        self.m = np.reshape(m_, self.size)

    # Override values from float array
    def override_values(self, val):
        # TODO : Optimize!!!!!!
        if val.shape != self.size:
            raise ValueError ("Matrix size not match")
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                v = floatToBits(val[i][j])
                # print("{} {:08b}  {:023b}".format(v>>31, v >> 23 & 0xff, v & 0x7fffff))
                self.e[i,j] = (v & 0x7f800000) >> 23
                # TODO : You know this is not good code, es, plz fix soon
                self.m[i,j] = ((v & 0x007fffff) >> (23 - self.mb + 1)) | (0x1 << (self.mb - 1))
                k = "-" if self.m[0,0] < 0 else "+"
                # print("{} {:08b} {:012b}".format(k, self.e[i,j], abs(self.m[i,j])))
                # print(self.e[i][j], format(self.m[i][j],"08b"))
                if (v >> 31) & 0x1:
                    self.m[i][j] = -self.m[i][j]
        # make groups later on
        # self.make_groups()

    def get_value(self, ind, hex=True):
        if self.m[ind] < 0:
            s = "-"
        else:
            s = "+"
        if hex:
            return "{}{:02x}_{}".format(s, self.e[ind], format(abs(self.m[ind]),"0{}x".format((self.mb//4))))
        else:
            return "{}{:02x}_{}".format(s, self.e[ind], format(abs(self.m[ind]),"0{}b".format(self.mb)))
    

    def get_value_float(self, ind):
        e_, m_ = self.e[ind], abs(self.m[ind])
        # Get the calculated matissa's bit
        #m_ = m_ >> 3
        t, c = m_, 0
        while (t != 0):
            t = t >> 1
            c += 1
        #print(format(e_,"08b"), format(m_,"08b"))
        m_ = (m_ << (self.mb - c + 1)) & mantissa_mask[self.mb]
        e_ += c - self.mb
        #print(format(e_,"08b"), format(m_,"08b"))
        if self.m[ind] < 0:
            return BitToFloats((1<<31)|(e_<<23)|(m_<<(23 - self.mb)))
        else:
            return BitToFloats((e_<<23)|(m_<<(23 - self.mb)))

    # Matrix multiplication
    def mm(self, val):
        if not isinstance(val, BFArray):
            raise ValueError ("Input is not an BFArray")
        
        # Suppose val is n*1 array
        if val.e.shape[1] != 1:
            raise ValueError ("Operation not supported")
        
        # Create tile of transposed value
        val_e_ = np.reshape(np.tile(np.transpose(val.e), self.size[0]),self.size)
        val_m_ = np.reshape(np.tile(np.transpose(val.m), self.size[0]),self.size)
        e_ = val_e_ + self.e - 127
        m_ = np.multiply(val_m_, self.m)

        # Finding max exponent on each line
        bs = np.zeros(self.size,dtype=np.intc)
        for i in range(self.size[0]):
            bs[i] = np.amax(e_[i])
        bs = bs - e_
        # Unify exponents and shift mantissa
        m_ = np.right_shift(m_,bs)
        # Add all mantissa        
        rm = m_.sum(axis=1)
        re = np.zeros((self.size[0], 1), dtype=np.intc)

        # Temporal Overrider
        out = BFArray((self.size[0], 1), mb = self.mb)
        # Match mantissa and apply shift
        for i in range(self.size[0]):
            # Get the mantiss'a bit
            # m_[i] is the value to handle. It should be self.mb + val.mb
            # And it should match to r.mb
            t, c = abs(rm[i]), 0
            while (t != 0):
                t = t >> 1
                c += 1
            # Shift mantissa bits
            out.m[i] = rm[i] >> (c - out.mb)
            re[i] = e_[i][0] + (c - self.mb - val.mb)
        out.e = re
        out.m = out.m
        return out