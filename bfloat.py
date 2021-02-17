"""
bfloat.py
Brain floating point python datatype with custom mantissa bits
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
# https://stackoverflow.com/questions/14431170/get-the-bits-of-a-float-in-python
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

def BitToFloatsP(f):
    print("{:20d}".format(f), format(f,"032b"))
    if f>>31 == 1:
        f = - (f ^ 0xffffffff)
    # TODO : make this work
    s = struct.pack('>l', f)
    return struct.unpack('>f', s)[0]

mantissa_mask = [     0x0
                ,     0x1,     0x3,     0x7,     0xf,    0x1f,    0x3f,    0x7f,    0xff
                ,   0x1ff,   0x3ff,   0x7ff,   0xfff,  0x1fff,  0x3fff,  0x7fff,  0xffff
                , 0x1ffff, 0x3ffff, 0x7ffff, 0xfffff,0x1fffff,0x3fffff,0x7fffff]

class BFloat(object):

    def __init__(self, val=1, mb=8):
        '''
        Brain floating-point format with custom mantissa
        similar as bfloat16, but data is handled as below
            sign(1-bit) : for sign
            exp(8-bit fix): exponential value between -126 ~ 127
            mantissa(custom bits): selective mantissa between 1~23
        '''
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
        elif isinstance(val, BFloat):
            self.s, self.e, self.m, self.mb = val.s, val.e, val.m, val.mb
            self.mask = mantissa_mask[self.mb]
        
        # Mask the mantissa
        self.m = self.m & self.mask

    # Type Conversion
    def __int__(self):
        return (self.s << 31) | (self.e << 23) | (self.m << (23 - self.mb))
    def __float__(self):
        # print("BitToFloats called, ",self.s, self.e, self.m)
        return BitToFloats((self.s << 31) | (self.e << 23) | (self.m << (23 - self.mb)))

    # Representors
    def __repr__(self):
        return str(self)
    def __str__(self):
        return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"0{}b".format(self.mb)))
    
    # Binary Operators
    # + (float, int, BFloat)
    def __add__ (self, value):
        '''
        Add operation between two BFloat numbers
        '''
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, mb=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        
        # Select the bigger value to match the exponent
        if value.e > self.e:
            bigN, smallN = value, self
        elif value.e < self.e:
            bigN, smallN = self, value
        else:
            if value.m >= self.m:
                bigN, smallN = value, self
            else:
                bigN, smallN = self, value

        # If one number is way too small to compute together, return bigger value
        eDiff = bigN.e - smallN.e
        if eDiff > bigN.mb:
            return bigN

        # New BFloat object to store new result, or it will override self or value
        r = BFloat()
        r.s, r.e, r.mb = bigN.s, bigN.e, max(bigN.mb, smallN.mb)
        if bigN.s ^ smallN.s == 1: # Subtract value
            # Very very rare case: same mantissa and same exponent, which should lead to zero
            if bigN.m == smallN.m and bigN.e == smallN.e:
                return BFloat(0)
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
    # - (float, int, BFloat)
    def __sub__ (self, value):
        '''
        Subtract operation between two BFloat numbers, which basically using add operation
        '''
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, mb=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        
        value_ = value
        value_.s = value.s ^ 0x1
        r = value_ + self
        return r
    # * (float, int, BFloat)
    def __mul__(self, value):
        '''
        # TODO : match with other exponent?
        '''
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, mb=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")

        r = BFloat(0, mb=max(self.mb, value.mb))
        if (self.m==0x0 and self.e==0x0) or (value.m==0x0 and value.e==0x0): # If one number is zero return zero
            return r

        # Sign
        r.s = self.s ^ value.s
        # Exponent
        r.e = self.e + value.e - 127
        # Mantissa
        r.m = (self.m | (0x1 << self.mb)) * (value.m | (0x1 << value.mb))

        # Have to shift value to match the first mantissa bit
        # Get the calculated matissa's bit
        t, c = r.m, 0
        while (t != 0):
            t = t >> 1
            c += 1
        # Shift mantissa and apply to exponential
        es = c - 1 - self.mb - value.mb
        r.e += es
        # Suppose always mantissa bit have to shift right
        r.m = r.m >> (value.mb+self.mb-r.mb + es)

        # Apply mask to remove unnessary mantissa bit
        r.m = r.m & r.mask
        return r
    # / (float, int, BFloat)
    def __truediv__(self, value):
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, mb=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")

        r = BFloat(0, mb=max(self.mb, value.mb))
        if (self.m==0x0 and self.e==0x0) or (value.m==0x0 and value.e==0x0): # If one number is zero return zero
            return r

        # Sign
        r.s = self.s ^ value.s
        # Exponent
        r.e = self.e - value.e + 127
        # Mantissa
        r.m = ((self.m | (0x1 << self.mb)) << r.mb) // (value.m | (0x1 << value.mb))
        # Have to shift value to match the first mantissa bit
        # Get the calculated matissa's bit
        t, c = r.m, 0
        while (t != 0):
            t = t >> 1
            c += 1
        # Shift mantissa and apply to exponential
        es = 1 + r.mb - c
        r.e -= es
        # Suppose always mantissa bit have to shift left
        r.m = r.m << es
        # Apply mask to remove unnessary mantissa bit
        r.m = r.m & r.mask
        
        return r
    # // __floordiv__ : not required
    # % __mod__ : not required
    # ** __pow__ : not required...?
    # >> __rshift__ : not required
    # << __lshift__ : not required
    # & (BFloat)
    def __and__(self, other):
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        r = BFloat(0, mb=self.mb)
        r.s, r.e, r.m = self.s & other.s, self.e & other.e, self.m & other.m
        return r
    # | (BFloat)
    def __or__(self, other):
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        r = BFloat(0, mb=self.mb)
        r.s, r.e, r.m = self.s | other.s, self.e | other.e, self.m | other.m
        return r 
    # ^ (BFloat)
    def __xor__(self, other):
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        r = BFloat(0, mb=self.mb)
        r.s, r.e, r.m = self.s ^ other.s, self.e ^ other.e, self.m ^ other.m
        return r 

    # Comparison Operators
    # TODO : Optimize comparison
    # < (float, int, BFloat)
    def __lt__(self, value):
        if isinstance(value, float) or isinstance(value, int):
            return float(self) < value
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        return float(self) < float(value)
    # > (float, int, BFloat)
    def __gt__(self, value):
        if isinstance(value, float) or isinstance(value, int):
            return float(self) > value
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        return float(self) > float(value)
    # <= (float, int, BFloat)
    def __le__(self, value):
        if isinstance(value, float) or isinstance(value, int):
            return float(self) <= value
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        return float(self) <= float(value)
    # >= (float, int, BFloat)
    def __ge__(self, value):
        if isinstance(value, float) or isinstance(value, int):
            return float(self) >= value
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        return float(self) >= float(value)
    # == (float, int, BFloat)
    def __eq__(self, value):
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, mb=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        return float(self) == float(value)
    # != (float, int, BFloat)
    def __ne__(self, value):
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, mb=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Input type not supported")
        return float(self) != float(value)
    
    # Unary Operators
    # -
    def __neg__(self):
        r = BFloat(self)
        r.s = r.s^1
        return r
    # +
    def __pos__(self):
        return BFloat(self)
    # ~
    def __invert__(self):
        r = BFloat(self)
        r.s, r.e, r.m = r.s^0x1, r.e^0xff, r.m^r.mask
        return r
    # abs
    def __abs__(self):
        r = BFloat(self)
        r.s = 0
        return float(r)

