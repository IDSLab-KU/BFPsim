##############################
# Logger 1.0 by R3C0D3r      #
# cryptographcode@gmail.com  #
##############################

from datetime import datetime
import os

rCol = '\033[0m'

tCol = dict()
tCol['k']  = tCol['black']          = '\033[30m'
tCol['r']  = tCol['red']            = '\033[31m'
tCol['g']  = tCol['green']          = '\033[32m'
tCol['y']  = tCol['yellow']         = '\033[33m'
tCol['b']  = tCol['blue']           = '\033[34m'
tCol['m']  = tCol['magenta']        = '\033[35m'
tCol['c']  = tCol['cyan']           = '\033[36m'
tCol['w']  = tCol['white']          = '\033[37m'
tCol['br'] = tCol['brightred']      = '\033[91m'
tCol['bg'] = tCol['brightgreen']    = '\033[92m'
tCol['by'] = tCol['brightyellow']   = '\033[93m'
tCol['bb'] = tCol['brightblue']     = '\033[94m'
tCol['bm'] = tCol['brightmagenta']  = '\033[95m'
tCol['bc'] = tCol['brightcyan']     = '\033[96m'
tCol['bw'] = tCol['brightwhite']    = '\033[97m'

bCol = dict()
bCol['k']  = bCol['black']          = '\033[40m'
bCol['r']  = bCol['red']            = '\033[41m'
bCol['g']  = bCol['green']          = '\033[42m'
bCol['y']  = bCol['yellow']         = '\033[43m'
bCol['b']  = bCol['blue']           = '\033[44m'
bCol['m']  = bCol['magenta']        = '\033[45m'
bCol['c']  = bCol['cyan']           = '\033[46m'
bCol['w']  = bCol['white']          = '\033[47m'
bCol['br'] = bCol['brightred']      = '\033[101m'
bCol['bg'] = bCol['brightgreen']    = '\033[102m'
bCol['by'] = bCol['brightyellow']   = '\033[103m'
bCol['bb'] = bCol['brightblue']     = '\033[104m'
bCol['bm'] = bCol['brightmagenta']  = '\033[105m'
bCol['bc'] = bCol['brightcyan']     = '\033[106m'
bCol['bw'] = bCol['brightwhite']    = '\033[107m'

def FloatToDatetime(fl):
    return datetime.fromtimestamp(fl)

def DatetimeToFloat(d):
    return d.timestamp()
class Logger:

    def __init__(self):
        self.timeInit = datetime.now().timestamp()
        self.isLogFile = False
        self.logFileLocation = None
        self.logFilePointer = None
        self.logValid = False
        self.printCurrentTime = True
        self.printElapsedTime = True
        self.MSPrecision = 3
        self.printLevel = 0
        self.messages = True
        self.Message("Logger: is initalized.")

    def SetLogFile(self, b, t=None):
        self.isLogFile = b
        if self.isLogFile:
            if t == None:
                if not os.path.exists("./logs"):
                    os.makedirs("./logs")
                self.logFileLocation = "./logs/%s.log"%(str(datetime.now())[:-7].replace("-","").replace(":","").replace(" ","_"))
            else:
                self.logFileLocation = t
            
    def SetPrintLevel(self, v):
        assert isinstance(v, int), "Logger: Inputed value is not int"
        self.printLevel = v
        self.Message("Logger: Print Level is set to %d."%v)
    
    def SetMSPrecision(self, v):
        assert v in [0, 1, 2, 3, 4, 5], "Logger: Precision Not Available"
        self.MSPrecision = v
        if self.MSPrecision > 5:
            self.MSPrecision = 5
        elif self.MSPrecision < 0:
            self.MSPrecision = 0
        self.Message("Logger: Milisecond Precision is set to %s."%v)

    def SetMessages(self, b):
        assert b in [True, False], "Logger: Incorrect value has been inputed"
        self.Message("Logger: Messages from logger is set to %s."%b, f=True)
        self.messages = b

    def SetPrintCurrentTime(self, b):
        assert b in [True, False], "Logger: Incorrect value has been inputed"
        self.printCurrentTime = b
        self.Message("Logger: PrintCurrentTime is set to %s."%b)

    def SetPrintElapsedTime(self, b):
        assert b in [True, False], "Logger: Incorrect value has been inputed"
        self.Message("Logger: PrintElapsedTime is set to %s."%b)
        self.printElapsedTime = b

    def GetCurrentTime(self):
        return str(datetime.now())[5:-3]

    def GetElapsedTime(self):
        et = datetime.now().timestamp() - self.timeInit
        etMS = int(et * 1000000) % 1000000
        etS = int(et % 60)
        etM = int((et / 60) % 60)
        etH = int(et / 3600)
        return "%02d:%02d:%02d.%6s"%(etH,etM,etS,str(etMS).zfill(6))

    def Message(self, msg, f=False):
        if self.messages or f:
            self.Print(msg,current=False,elapsed=False,col='k',bg='bc',file=False)

    def Print(self, msg, level = 0, current = None, elapsed = None, col='', bg='', end = '\n', file = True, flush = False):
        if level <= self.printLevel:
            t = ""
            if self.MSPrecision == 0:
                d = -7
            else:
                d = self.MSPrecision - 6
            if current == None:
                current = self.printCurrentTime
            if elapsed == None:
                elapsed = self.printElapsedTime
            if current:
                t += str(datetime.now())[5:][:d]
            if elapsed:
                t += "[" + self.GetElapsedTime()[:d] + "]"
            if current or elapsed:
                t += ":"
            if self.isLogFile and file: # Log file doesn't record colors
                if self.logFilePointer == None:
                    self.logFilePointer = open(self.logFileLocation, mode="w", newline='', encoding='utf-8')
                    self.Message("Logger: Creating log file on %s."%self.logFileLocation)
                self.logFilePointer.write(t + msg + end)

            if col != '':
                t += tCol[col]
            if bg != '':
                t += bCol[bg]
            t += msg + rCol
            e, f = end, flush
            print(t, end = e, flush = f)

Log = Logger()
