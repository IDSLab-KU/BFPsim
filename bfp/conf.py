"""
    This code is part of the BFPSim (https://github.com/ids-Lab-DGIST/BFPSim)

    Seunghyun Lee (R3C0D3r) from IDSLab, DGIST
    coder@dgist.ac.kr

    License: CC BY 4.0
"""

# "wi, 36" returns (4,1,3,3)
# [1,4,3,3] returns original stuffs
# single number with 1 returns (1,1,1,1)
def GetDimension(val, dim=4):
    if type(val) == list:
        return tuple(val)
    elif type(val) == str:
        v = val.replace(" ","").split(",")
        gs = int(v[1])
        if dim == 4:
            if v[0].lower() == "wi":
                return (gs//9,1,3,3)
            elif v[0].lower() == "wo":
                return (1,gs//9,3,3)
            elif v[0].lower() == "fx":
                return (1,1,gs//3,3)
            elif v[0].lower() == "fy":
                return (1,1,3,gs//3)
            elif v[0].lower() == "fc":
                return (1,gs//9,3,3)
            else:
                ValueError("BFPConf direction({v[0]}) is not recognized for tensor dim of {dim}")
        elif dim == 3: # Other dimension
            return (gs, 1, 1)
        elif dim == 2:
            return (gs, 1)
        else:
            ValueError("BFPConf tensor dim not supported: {dim}")
    elif val == 1:
        if dim == 4:
            return (1,1,1,1)
        elif dim == 3:
            return (1,1,1)
        elif dim == 2:
            return (1,1)

def GetTupleShortString(t):
    s = "("
    for i in t:
        s +="%d,"%i
    s = s[:-1]
    s += ")"
    return s

class BFPConf():
    def __init__(self, dic=None, bwg_boost = 1.0):
        if dic == None:
            dic = dict()
        self.type   = dic["type"]                   if "type"   in dic.keys() else "Conv2d"
        if self.type.lower() == "conv2d":
            di, dw = 4, 4
        elif self.type.lower() == "linear":
            di, dw = 2, 2

        # Foward - Weight
        self.fw     = dic["fw"]                         if "fw"     in dic.keys() else True
        self.fw_bit = dic["fw_bit"]                     if "fw_bit" in dic.keys() else 8
        self.fw_dim  = GetDimension(dic["fw_dim"],di)   if "fw_dim" in dic.keys() else (16,1,3,3)

        # Forward - Input
        self.fi     = dic["fi"]                         if "fi"     in dic.keys() else True
        self.fi_bit = dic["fi_bit"]                     if "fi_bit" in dic.keys() else self.fw_bit
        self.fi_dim  = GetDimension(dic["fi_dim"],dw)   if "fi_dim" in dic.keys() else (4,1,3,3)

        # Forward - Output
        self.fo     = dic["fo"]                         if "fo"     in dic.keys() else False
        self.fo_bit = dic["fo_bit"]                     if "fo_bit" in dic.keys() else self.fw_bit
        self.fo_dim  = GetDimension(dic["fo_dim"],di)   if "fo_dim" in dic.keys() else (1,1,1,1)

        # Backward - Output gradient while calculating input gradient
        self.bio     = dic["bio"]                       if "bio"     in dic.keys() else True
        self.bio_bit = dic["bio_bit"]                   if "bio_bit" in dic.keys() else self.fi_bit
        self.bio_dim  = GetDimension(dic["bio_dim"],di) if "bio_dim" in dic.keys() else self.fi_dim

        # Backward - Weight while calculating input gradient
        self.biw     = dic["biw"]                       if "biw"     in dic.keys() else True
        self.biw_bit = dic["biw_bit"]                   if "biw_bit" in dic.keys() else self.fw_bit
        self.biw_dim  = GetDimension(dic["biw_dim"],dw) if "biw_dim" in dic.keys() else self.fw_dim

        # Backward - Calculated input gradient
        self.big     = dic["big"]                       if "big"     in dic.keys() else False
        self.big_bit = dic["big_bit"]                   if "big_bit" in dic.keys() else self.fi_bit
        self.big_dim  = GetDimension(dic["big_dim"],di) if "big_dim" in dic.keys() else (1,1,1,1)

        # Backward - Output gradient while calculating weight gradient
        self.bwo     = dic["bwo"]                       if "bwo"     in dic.keys() else True
        self.bwo_bit = dic["bwo_bit"]                   if "bwo_bit" in dic.keys() else self.fi_bit
        self.bwo_dim  = GetDimension(dic["bwo_dim"],di) if "bwo_dim" in dic.keys() else self.fi_dim

        # Backward - Input while calculating weight gradient
        self.bwi     = dic["bwi"]                       if "bwi"     in dic.keys() else True
        self.bwi_bit = dic["bwi_bit"]                   if "bwi_bit" in dic.keys() else self.fi_bit
        self.bwi_dim  = GetDimension(dic["bwi_dim"],di) if "bwi_dim" in dic.keys() else self.fi_dim

        # Backward - Calculated weight gradient
        self.bwg     = dic["bwg"]                       if "bwg"     in dic.keys() else False
        self.bwg_bit = dic["bwg_bit"]                   if "bwg_bit" in dic.keys() else self.fw_bit
        self.bwg_dim  = GetDimension(dic["bwg_dim"],dw) if "bwg_dim" in dic.keys() else (4, 1, 3, 3)

        self.bwg_boost = bwg_boost

    def __repr__(self):
        return str(self)
    def __str__(self):
        s = "["
        s += "FW/" if self.fw else "  /" 
        s += "FI/" if self.fi else "  /" 
        s += "FO/" if self.fo else "  /" 
        s += "BIO/" if self.bio else "   /" 
        s += "BIW/" if self.biw else "   /" 
        s += "BIG/" if self.big else "   /" 
        if (self.bwo_bit == self.bio_bit and self.bwo_dim == self.bio_dim):
            s += "BWO*/" if self.bwo else "   /" 
        else:
            s += "BWO/" if self.bwo else "   /" 
        if (self.bwi_bit == self.fi_bit and self.bwi_dim == self.fi_dim):
            s += "BWI*/" if self.bwi else "   /" 
        else:
            s += "BWI/" if self.bwi else "   /" 
        s += "BWG]" if self.bwg else "   ]"
        s += ",bit="
        if self.fw_bit == self.fi_bit == self.fo_bit == self.bio_bit == self.biw_bit == self.big_bit == self.bwo_bit == self.bwi_bit == self.bwg_bit:
            s += '{}'.format(self.fw_bit)
        else:
            s += "[{}/".format(self.fw_bit) if self.fw else "[_/" 
            s += "{}/".format(self.fi_bit) if self.fi else "_/" 
            s += "{}/".format(self.fo_bit) if self.fo else "_/" 
            s += "{}/".format(self.bio_bit) if self.bio else "_/" 
            s += "{}/".format(self.biw_bit) if self.biw else "_/" 
            s += "{}/".format(self.big_bit) if self.big else "_/" 
            s += "{}/".format(self.bwo_bit) if self.bwo else "_/" 
            s += "{}/".format(self.bwi_bit) if self.bwi else "_/" 
            s += "{}".format(self.bwg_bit)  if self.bwg else "_]"
        s += ",dim="
        if self.fw_dim == self.fi_dim == self.fo_dim == self.bio_dim == self.biw_dim == self.big_dim == self.bwo_dim == self.bwi_dim == self.bwg_dim:
            s += '{}'.format(self.fw_dim)
        else:
            s += "[{}/".format(GetTupleShortString(self.fw_dim)) if self.fw else "[_/" 
            s += "{}/".format(GetTupleShortString(self.fi_dim)) if self.fi else "_/" 
            s += "{}/".format(GetTupleShortString(self.fo_dim)) if self.fo else "_/" 
            s += "{}/".format(GetTupleShortString(self.bio_dim)) if self.bio else "_/" 
            s += "{}/".format(GetTupleShortString(self.biw_dim)) if self.biw else "_/" 
            s += "{}/".format(GetTupleShortString(self.big_dim)) if self.big else "_/" 
            s += "{}/".format(GetTupleShortString(self.bwo_dim)) if self.bwo else "_/" 
            s += "{}/".format(GetTupleShortString(self.bwi_dim)) if self.bwi else "_/" 
            s += "{}".format(GetTupleShortString(self.bwg_dim))  if self.bwg else "_]"
        if self.bwg_boost != 1.0:
            s += ",bwg_boost={}".format(self.bwg_boost)
        return s
