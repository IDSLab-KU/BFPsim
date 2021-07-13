

class BFPConf():
    def __init__(self, dic=None, bwg_boost = 1.0):
        if dic == None:
            dic = dict()
        self.type   = dic["type"]              if "type"   in dic.keys() else "Conv2d"

        # Foward - Weight
        self.fw     = dic["fw"]                if "fw"     in dic.keys() else True
        self.fw_bit = dic["fw_bit"]            if "fw_bit" in dic.keys() else 8
        self.fw_dim  = tuple(dic["fw_dim"])    if "fw_dim" in dic.keys() else 36

        # Forward - Input
        self.fi     = dic["fi"]                if "fi"     in dic.keys() else True
        self.fi_bit = dic["fi_bit"]            if "fi_bit" in dic.keys() else self.fw_bit
        self.fi_dim  = tuple(dic["fi_dim"])    if "fi_dim" in dic.keys() else self.fw_dim

        # Forward - Output
        self.fo     = dic["fo"]                if "fo"     in dic.keys() else False
        self.fo_bit = dic["fo_bit"]            if "fo_bit" in dic.keys() else self.fw_bit
        self.fo_dim  = tuple(dic["fo_dim"])    if "fo_dim" in dic.keys() else self.fw_dim

        # Backward - Output gradient while calculating input gradient
        self.bio     = dic["bio"]                if "bio"     in dic.keys() else True
        self.bio_bit = dic["bio_bit"]            if "bio_bit" in dic.keys() else self.fo_bit
        self.bio_dim  = tuple(dic["bio_dim"])    if "bio_dim" in dic.keys() else self.fo_dim

        # Backward - Weight while calculating input gradient
        self.biw     = dic["biw"]                if "biw"     in dic.keys() else True
        self.biw_bit = dic["biw_bit"]            if "biw_bit" in dic.keys() else self.fw_bit
        self.biw_dim  = tuple(dic["biw_dim"])    if "biw_dim" in dic.keys() else self.fw_dim

        # Backward - Calculated input gradient
        self.big     = dic["big"]                if "big"     in dic.keys() else False
        self.big_bit = dic["big_bit"]            if "big_bit" in dic.keys() else self.fi_bit
        self.big_dim  = tuple(dic["big_dim"])    if "big_dim" in dic.keys() else self.fi_dim

        # Backward - Output gradient while calculating weight gradient
        self.bwo     = dic["bwo"]                if "bwo"     in dic.keys() else True
        self.bwo_bit = dic["bwo_bit"]            if "bwo_bit" in dic.keys() else self.fo_bit
        self.bwo_dim  = tuple(dic["bwo_dim"])    if "bwo_dim" in dic.keys() else self.fo_dim

        # Backward - Input while calculating weight gradient
        self.bwi     = dic["bwi"]                if "bwi"     in dic.keys() else True
        self.bwi_bit = dic["bwi_bit"]            if "bwi_bit" in dic.keys() else self.fi_bit
        self.bwi_dim  = tuple(dic["bwi_dim"])    if "bwi_dim" in dic.keys() else self.fi_dim

        # Backward - Calculated weight gradient
        self.bwg     = dic["bwg"]                if "bwg"     in dic.keys() else False
        self.bwg_bit = dic["bwg_bit"]            if "bwg_bit" in dic.keys() else self.fw_bit
        self.bwg_dim  = tuple(dic["bwg_dim"])    if "bwg_dim" in dic.keys() else self.fw_dim

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
            s += "[{}/".format(self.fw_dim) if self.fw else "[_/" 
            s += "{}/".format(self.fi_dim) if self.fi else "_/" 
            s += "{}/".format(self.fo_dim) if self.fo else "_/" 
            s += "{}/".format(self.bio_dim) if self.bio else "_/" 
            s += "{}/".format(self.biw_dim) if self.biw else "_/" 
            s += "{}/".format(self.big_dim) if self.big else "_/" 
            s += "{}/".format(self.bwo_dim) if self.bwo else "_/" 
            s += "{}/".format(self.bwi_dim) if self.bwi else "_/" 
            s += "{}".format(self.bwg_dim)  if self.bwg else "_]"
        s += ",bwg_boost={}".format(self.bwg_boost)
        return s
