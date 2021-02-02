###############################
# Logger
###############################

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

    def Exit(self):
        if self.isLogFile:
            self.logFilePointer.close()
            if not self.logValid:
                if os.path.exists(self.logFileLocation):
                    os.remove(self.logFileLocation)


    def SetLogFile(self, b, t=None):
        self.isLogFile = b
        if self.isLogFile:
            if t == None:
                self.logFileLocation = "./logs/%s.log"%(str(datetime.now())[:-7].replace("-","").replace(":","").replace(" ","_"))
            else:
                self.logFileLocation = t
            if not os.path.exists("./logs"):
                os.makedirs("./logs")
            

    def SetPrintCurrentTime(self, b):
        self.printCurrentTime = b

    def SetPrintElapsedTime(self, b):
        self.printElapsedTime = b

    def GetCurrentTime(self):
        return str(datetime.now())[5:-3]

    def GetElapsedTime(self):
        et = datetime.now().timestamp() - self.timeInit
        etMS = int(et * 1000) % 1000
        etS = int(et % 60)
        etM = int((et / 60) % 60)
        etH = int(et / 3600)
        return "%02d:%02d:%02d.%3d"%(etH,etM,etS,etMS)


    def Print(self, msg, current = True, elapsed = True, col='', bg='', end = '\n'):
        t = ""
        if current or (not current and self.printCurrentTime):
            t += str(datetime.now())[5:-3]
        if elapsed or (not elapsed and self.printElapsedTime):
            t += "[" + self.GetElapsedTime() + "]"
        if current or elapsed:
            t += ":"
        if self.isLogFile: # Log file doesn't record colors
            if self.logFilePointer == None:
                self.logFilePointer = open(self.logFileLocation, mode="w", newline='', encoding='utf-8')
            self.logFilePointer.write(t + msg + end)
            
        if col != '':
            t += tCol[col]
        if bg != '':
            t += bCol[bg]
        t += msg + rCol
        e = end

        print(t, end = e)

Log = Logger()