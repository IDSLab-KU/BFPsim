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

mantissa_mask = [     0x0
                ,     0x1,     0x3,     0x7,     0xf,    0x1f,    0x3f,    0x7f,    0xff
                ,   0x1ff,   0x3ff,   0x7ff,   0xfff,  0x1fff,  0x3fff,  0x7fff,  0xffff
                , 0x1ffff, 0x3ffff, 0x7ffff, 0xfffff,0x1fffff,0x3fffff,0x7fffff]

import numpy as np

from bfloat import BFloat

# BFloat but with not precise mantissa
class BFElement(BFloat):
    def __init__(self, val=0, mb=8):
        if isinstance(val, float) or isinstance(val, int):
            v = floatToBits(val)
            # Sign
            self.s = (v >> 31) & 0x1
            # Exponent
            self.e = (v & 0x7f800000) >> 23
            # Mantissa, with custom precise
            assert mb < 24 and mb > 0, "Mantissa bit not supported"
            self.m = (v & 0x007fffff) >> (23 - mb)
            self.mb = mb
            self.mask = mantissa_mask[mb]
            
            # VS: virtual shift : shift bytes to match exponent
            self.vs = 0
        elif isinstance(val, BFElement):
            self.s, self.e, self.m, self.mb, self.vs = val.s, val.e, val.m, val.mb, val.vs
            self.mask = mantissa_mask[self.mb]
        
        # Mask the mantissa
        self.m = self.m & self.mask
    
    # Type Conversion
    def __int__(self):
        return (self.s << 31) | (self.e << 23) | (self.m << (23 - self.mb))
    def __float__(self):
        # print("BitToFloats called, ",self.s, self.e, self.m)
        return BitToFloats((self.s << 31) | (self.e << 23) | (self.m << (23 - self.mb)))
    

    # TODO : Edit add and mul can process 

    def __add__ (self, value):
        '''
        Add operation between two BFloat numbers
        '''
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFElement(value, mb=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")

        # Overloading virtual shifts
        _self  = BFElement(self, mb=self.mb)
        _self.m = self.m - (self.m & mantissa_mask[_self.vs])
        _value = BFElement(value, mb=self.mb)
        _value.m = value.m - (value.m & mantissa_mask[_value.vs])
        
        # Select the bigger value to match the exponent
        if _value.e > _self.e:
            bigN, smallN = _value, _self
        elif _value.e < _self.e:
            bigN, smallN = _self, _value
        else:
            if _value.m >= _self.m:
                bigN, smallN = _value, _self
            else:
                bigN, smallN = _self, _value

        # If one number is way too small to compute together, return bigger value
        eDiff = bigN.e - smallN.e
        if eDiff > bigN.mb:
            return bigN

        # New BFloat object to store new result, or it will override self or value
        r = BFElement()
        r.s, r.e, r.mb = bigN.s, bigN.e, max(bigN.mb, smallN.mb)
        if bigN.s ^ smallN.s == 1: # Subtract value
            # Very very rare case: same mantissa and same exponent, which should lead to zero
            if bigN.m == smallN.m and bigN.e == smallN.e:
                return BFElement(0)
            # Calculate mantissa
            r.m = (bigN.m | (0x1 << bigN.mb)) - ((smallN.m | (0x1 << smallN.mb)) >> eDiff)
            # print("Mantissa: {} -> {} - {} -> {} = {}".format(bigN.m,(bigN.m | (0x1 << bigN.mb)),smallN.m,(smallN.m + (0x1 << smallN.mb)) >> eDiff,r.m))
            # TODO : Improve algorithm to not just ignoring lower bits, and actually calculate the bits
            #        This case, sometimes smaller mantissa bits maybe ignored. maybe applying shift later will work...?
            if r.m != 0:
                # If mantissa is not zero, decrease the exponent and shift mantissa if needed
                t, c = r.m, 0
                # Get the calculated matissa's bit
                while (t != 0):
                    t = t >> 1
                    c += 1
                # Decrease n exponent value to make first bit of mantissa to be 1
                if c < 1 + r.mb:
                    r.m = r.m << (1 + r.mb - c)
                    r.e -= (1 + r.mb - c)
            else:
                # If mantissa is zero, decrease the exponent as small as possible
                r.e = r.e << (1 + r.mb)
        else: # Addition
            r.m = (bigN.m | (0x1 << bigN.mb)) + ((smallN.m | (0x1 << smallN.mb)) >> eDiff)
            # Increase one exponent value if a mantissa is bigger than mantissa bit
            # It only happens when adding numbers
            if r.m >> (self.mb+1) > 0:
                r.m = r.m >> 1
                r.e += 1

        # Apply mask to remove unnessary mantissa bit
        r.m = r.m & self.mask
        return r

    def __mul__(self, value):
        '''
        # TODO : match with other exponent?
        '''
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFElement(value, mb=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")

        # Overloading virtual shifts
        _self  = BFElement(self, mb=self.mb)
        _self.m = self.m - (self.m & mantissa_mask[_self.vs])
        _value = BFElement(value, mb=self.mb)
        _value.m = value.m - (value.m & mantissa_mask[_value.vs])

        r = BFElement(0, mb=max(_self.mb, _value.mb))
        if (_self.m==0x0 and _self.e==0x0) or (_value.m==0x0 and _value.e==0x0): # If one number is zero return zero
            return r

        # Sign
        r.s = _self.s ^ _value.s
        # Exponent
        r.e = _self.e + _value.e - 127
        # Mantissa
        r.m = (_self.m | (0x1 << _self.mb)) * (_value.m | (0x1 << _value.mb))

        # Have to shift value to match the first mantissa bit
        # Get the calculated matissa's bit
        t, c = r.m, 0
        while (t != 0):
            t = t >> 1
            c += 1
        # Shift mantissa and apply to exponential
        es = c - 1 - _self.mb - _value.mb
        r.e += es
        # Suppose always mantissa bit have to shift right
        r.m = r.m >> (_value.mb+_self.mb-r.mb + es)

        # Apply mask to remove unnessary mantissa bit
        r.m = r.m & r.mask
        return r
    
    # Representors
    def __repr__(self):
        return str(self)
    def __str__(self):
        # Print virtual exponent and virtual mantissa
        e = self.e + self.vs
        m = (self.m | (0x1 << self.mb)) >> (1 + self.vs)
        s = '-' if self.s else '+'
        return '{}{}_{}'.format(s, format(e,"02x"), format(m,"0{}x".format((self.mb-1)//4+1)))
        
class BFArray(object):
    def __init__(self, size, mb=8, group_size=36):
        '''
        size: Size of array provided with tuple
        mb: Size of mantissa bits
        '''
        # Size of the array
        # On linear array, it's (output, input)
        self.size = size
        self.count = 1
        for i in self.size:
            self.count *= i
        assert mb < 24 and mb > 0, "Mantissa bit not supported"
        self.mb = mb

        self.group_size = group_size # 36

        # Data stored as 1d-array for improved speed
        # Only supports 2d-like
        self.data = [None] * self.count

    # Initialize data with grouping numbers
    def initialize_data(self, method=0):
        # for i in range(self.size[0]*self.size[1]):
        t = np.random.randn(self.count)
        for i in range(self.count):
            self.data[i] = BFElement(t[i])
        # print(self.data)
        self.make_groups()

    def make_groups(self):
        # Groups weights
        mi, me = 0, 0
        for i in range(self.count):
            if self.data[i].e > me:
                mi = i
                me = self.data[i].e
            # set group's virtual exponents
            if (i+1)%self.group_size == 0:
                si = i//self.group_size*self.group_size
                for i in range(si, si+self.group_size):
                    self.data[i].vs = me - self.data[i].e
                mi, me = 0, 0

        # Stage 1: find maximum exponent on group

    def D(self, index):
        # Only supports 2d-like
        return self.data[index[1]+index[0]*self.size[1]]

    def __str__(self):
            return 'BFArray Object size of {}'.format(self.size)

    # *, Matrix Multiplication between two BFArray Objects
    def __mul__(self, value):
        if not isinstance(value, BFArray):
            raise ValueError ("Input is not an BFArray")
        # print("Matrix Multiplication")
        

        # Only works with 2dim - 2dim
        if value.size[1] != self.size[0]:
            raise ValueError ("Dim not match: Input {}, Output {}".format(self.size, value.size))
        # TODO : Override arguments
        r = BFArray((value.size[0], self.size[1]))

        # Do the computation
        for i0 in range(value.size[0]):
            for i1 in range(self.size[1]):
                _r = BFElement(0.0)
                for i2 in range(self.size[0]):
                    _r += self.D((i2,i1)) * value.D((i0,i2))
                # How to group the result...?
                r.data[i0 + i1*r.size[0]] = _r
        return r