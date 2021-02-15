
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

class BFloat(object):

    def __init__(self, val=1, m=8):
        '''
        Brain floating-point format with custom mantissa
        similar as bfloat16, but data is handled as below
            sign(1-bit) : for sign
            exp(8-bit fix): exponential value between -126 ~ 127
            mantissa(custom bits): mantissa select between 4, 8, 16
                mantissa should be originally 7 bits. but since 
        '''

        v = floatToBits(val)

        # Sign
        self.s = (v >> 31) & 0x1

        # Exponent
        self.e = (v & 0x7f800000) >> 23

        # Mantissa, with custom precise
        assert m in [4, 7, 8, 16], "Mantissa bit not supported"
        self.m = (v & 0x007fffff) >> 23 - m
        self.mb = m
        if m == 4:
            self.mask = 0xf
        elif m == 7:
            self.mask = 0x7f
        elif m == 8:
            self.mask = 0xff
        elif m == 16:
            self.mask = 0xffff
        self.m = self.m & self.mask

    def __float__(self):
        # print("BitToFloats called, ",self.s, self.e, self.m)
        return BitToFloats((self.s << 31) | (self.e << 23) | (self.m << (23 - self.mb)))

    # Represent the bits by divided
    def __repr__(self):
        if self.mb == 4:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"04b"))
        elif self.mb == 7:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"07b"))
        elif self.mb == 8:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"08b"))
        elif self.mb == 16:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"016b"))
        else:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"023b"))
    
    def __str__(self):
        if self.mb == 4:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"04b"))
        elif self.mb == 7:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"07b"))
        elif self.mb == 8:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"08b"))
        elif self.mb == 16:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"016b"))
        else:
            return '{} {} {}'.format(self.s, format(self.e,"08b"), format(self.m,"023b"))
    
    # Multiplication
    def __mul__(self, value):
        '''
        # TODO : match with other exponent?
        '''
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, m=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Not supported computation")
        # Do the math
        self.e > value.e
        return 0
    
    # Addition
    def __add__ (self, value):
        '''
        Add operation between two BFloat numbers
        '''
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, m=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Not supported computation")
        
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
            # Decrease n exponent value to make first bit of mantissa to be 1
            if (r.m >> (bigN.mb)) != 0x1 and r.m != 0: # There is very rare case that mantissa is zero, which may leads to infinite loop...
                while (r.m >> (bigN.mb)) != 0x1:
                    r.m = r.m << 1
                    r.e -= 1
        else: # Addition
            r.m = (bigN.m | (0x1 << bigN.mb)) + ((smallN.m | (0x1 << smallN.mb)) >> eDiff)
            # Increase one exponent value if a mantissa is bigger than mantissa bit
            # It only happens when adding numbers
            if r.m >> (self.mb+1) > 0:
                r.m = r.m >> 1
                r.e += 1

        # Remove the useless bits(especially the 9th or 17th mantissa bit), becuase it should be
        r.m = r.m & self.mask
        return r
    
    # Subtraction
    def __sub__ (self, value):
        '''
        Subtract operation between two BFloat numbers, which basically using add operation
        '''
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, m=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Not supported computation")
        
        value_ = value
        value_.s = value.s ^ 0x1
        r = value_ + self
        return r

    # Comparison
    def _cmp (self, value):
        '''
        Compare two values
        '''
        # If the inputed value is float or int, create the BFloat and compute, with same mantissa
        if isinstance(value, float) or isinstance(value, int):
            value = BFloat(value, m=self.mb)
        if not isinstance(value, BFloat):
            raise ValueError ("Not supported computation")
    
    def __gt__ (self, value):
        pass



def TestBasic(v, m=8):
    print("Inputed FP32:      ", v)
    print("Bit Representation:", format(floatToBits(v),"032b"))
    bf = BFloat(v, m)
    print("Converted BFloat{:2d}: {}".format(m, bf))
    print("Value of BFloat:   ", float(bf))
    print("Difference:        ", float(bf)-v)

def TestAdd(v1, v2, m=8):
    b1, b2 = BFloat(v1), BFloat(v2)
    print("First number:  {:10.5f} -> {} {:10.5f}".format(v1, b1, float(b1)))
    print("Second number: {:10.5f} -> {} {:10.5f}".format(v2, b2, float(b2)))
    print("Add:                         {} {:10.5f}".format(b1+b2, float(b1+b2)))
    print("Difference:    {:10.5f}                        {:10.5f}".format(v1+v2, v1+v2 - float(b1+b2)))
    print()

def TestSub(v1, v2, m=8):
    b1, b2 = BFloat(v1), BFloat(v2)
    print("First number:  {:10.5f} -> {} {:10.5f}".format(v1, b1, float(b1)))
    print("Second number: {:10.5f} -> {} {:10.5f}".format(v2, b2, float(b2)))
    print("Sub:                         {} {:10.5f}".format(b1-b2, float(b1-b2)))
    print("Difference:    {:10.5f}                        {:10.5f}".format(v1-v2, v1-v2 - float(b1-b2)))
'''
TestBasic(-3.1415926535)
TestBasic(3.1415926535)
TestBasic(-33.1415926535)
TestAdd(1.414592, 10.1922)
TestAdd(-1.414592, 10.1922)
TestAdd(1.414592, -10.1922)
TestAdd(-1.414592, -10.1922)
TestSub(1.414592, 10.1922)
TestSub(-1.414592, 10.1922)
TestSub(1.414592, -10.1922)
TestSub(-1.414592, -10.1922)
'''

import math

import numpy as np

ERROR_RATE = 1
ERROR_VALUE = 0.1

def TestRepeat(t="add", n=10000, p=1000, mb1=8, mb2=8, slient=False):
    a = np.random.randn(n)
    b = np.random.randn(n)
    error, errorP = 0.0, 0.0
    fail, miss, success = 0, 0, 0
    if t == "add":
        ts = "+"
    elif t == "sub":
        ts = "-"
    for i in range(n):
        b1, b2 = BFloat(a[i], mb1), BFloat(b[i], mb2)
        if t == "add":
            br = b1+b2
            r = a[i]+b[i]
        elif t == "sub":
            br = b1-b2
            r = a[i]-b[i]
        diff = abs(r - float(br))
        diffP = abs(diff / (r))
        # diff = 0
        if math.isnan(float(br)) or math.isnan(diff) or abs(diff) > 1000: # Absoulte error... something got really wrong
            if not slient:
                print("FAIL @ {}\n{}{}{}={} \n  -> {}{}{}={} | {}".format(i+1,a[i],ts,b[i],r,float(b1),ts,float(b2),float(br),diff))
                print(b1,b2,br)
            fail += 1
        elif diffP > ERROR_RATE and diff > ERROR_VALUE: # Minor miss, error over 1% with value of 0.1, maybe because of one or two exponent bits
            if not slient:
                print("MISS @ {}\n{}{}{}={} \n  -> {}{}{}={} | {}".format(i+1,a[i],ts,b[i],r,float(b1),ts,float(b2),float(br),diff))
                BitToFloatsP((b1.s << 31) | (b1.e << 23) | (b1.m << (23 - b1.mb)))
                print(b1.s, b1.e, b1.m)
                print(b1,b2,br)
            # print("{:6.3f}+{:6.3f}={:6.3f} -> {:6.3f}+{:6.3f}={:6.3f} | {:6.3f}".format(a[i],b[i],r,float(b1),float(b2),float(b1+b2),diff))
            miss += 1
        else:
            success += 1
            error += diff
            errorP += diffP
        if p!=0 and (i+1)%p == 0:
            print("{:8d}/{:8d} Error (Only Success):{:10.6f} ({:7.3f}%)".format(i+1,n,error/(success),errorP/(success)*100))
    
    print("Test on {:s} finished, mantissa bit {}, {}".format(t, mb1, mb2))
    print("  Total:   {:10d}\n  Success: {:10d}({:6.2f})\n  Failed:  {:10d}({:6.2f})\n  Missed:  {:10d}({:6.2f})".format(n, success, success/n*100, fail, fail/n*100, miss, miss/n*100))
    print("Average Error on Success:{:10.6f} ({:7.3f}%)".format(error/(success),errorP/(success)*100))
    print()

for i in ["add", "sub"]:
    TestRepeat(i, 1000000,100000, mb1=4, mb2=4, slient=True)