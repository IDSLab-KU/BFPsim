


COMP_TYPE = {
    "fi" :  0,
    "fw" :  1,
    "fo" : 2,
    "bio" : 10,
    "biw" : 11,
    "big" : 12,
    "bwo" : 20,
    "bwi" : 21,
    "bwg" : 22
}

class Flags:
    def __init__(self) -> None:
        self.ZSE = False
        self.DEBUG = False

FLAGS = Flags()

CUDA_THREADSPERBLOCK = 1024