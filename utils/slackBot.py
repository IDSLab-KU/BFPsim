# import logging
# logging.basicConfig(level=logging.DEBUG)
import os
import time
import socket
import datetime

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

def TimeStr(ts):
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def TimeStrH(ts):
    s = "%02d:%02d:%02d"%(ts//3600,(ts//60)%60,ts%60)
    return s

class slackBot_:
    def __init__(self) -> None:
        self.client = None
        self.minMessageInterval = 5
        self.latestMessageTime = time.time() / 1000000 - self.minMessageInterval
        self.isLimit = True
        self.channel = "#server_bot"

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.hostName = s.getsockname()[0]
            print("slackBot: Detected server location: %s"%self.hostName)
            s.close()
        except:
            print("slackBot: Couldn't find server ip")
            self.hostName = "Unknown"
        self.pid = str(os.getpid())
        self.processInfo = self.hostName + ":" + self.pid
        
        self.startTime = time.time()

    def SetToken(self, token):
        self.client = WebClient(token=token)

    def SetProcessInfo(self, name):
        self.processInfo = name

    def DisableLimit(self):
        self.isLimit = False
    
    def EnableLimit(self):
        self.isLimit = True

    def SetChannel(self, channel):
        self.channel = channel
        self.Send(":eyes: This channel is set to bot message channel.")

    def SendProgress(self, progress, estimated = True, length = 20, message="", channel=""):
        if progress < 0 or progress > 1:
            print("slackBot ERROR: Progress is not between 0 and 1")

        s = ":arrow_right: `%s` Progress : `%3.2f%%`\n"%(self.processInfo, progress*100)
        for i in range(length):
            if (i+1)/length <= progress:
                s += ":black_large_square:"
            else:
                s += ":black_small_square:"
        if length > 0:
            s += "\n"
        if estimated:
            currentTime = time.time()
            executed = currentTime - self.startTime
            estimated = self.startTime + executed / progress

            s += "Elapsed `%s`, Expected finishing time `%s` (`%s` Left)\n"%(TimeStrH(executed),TimeStr(estimated), TimeStrH(estimated-time.time()))
        if message != "":
            s += "Additional Message:\n"
        self.Send(message, s, channel)

    def SendStartSignal(self, message="", channel=""):
        s = ":large_green_circle: `%s` Started @ `%s` \n"%(self.processInfo, TimeStr(time.time()))
        if message != "":
            s += "Additional Message:\n"      
        self.Send(message, s, channel)

    def SendEndSignal(self, message="", channel=""):
        s = ":ballot_box_with_check: `%s` Finished! @ `%s` \n"%(self.processInfo, TimeStr(time.time()))
        if message != "":
            s += "Additional Message:\n"   
        self.Send(message, s, channel)

    def SendError(self, message, channel=""):
        self.Send(message, ":no_entry: `%s` Error!\n"%self.processInfo, channel)

    def SendWarning(self, message, channel=""):
        self.Send(message, ":warning: `%s` Warning...\n"%self.processInfo, channel)

    def SendMessage(self, message, channel=""):
        self.Send(message, ":pencil: `%s` Message\n"%self.processInfo, channel)

    def SendPing(self):
        self.Send()

    def Send(self, message="", prefix="", channel=""):
        # Check client is defined
        if self.client == None:
            print("slackBot ERROR: Please set token using slackBot.setToken(token)")
            return
        # Check if min message interval is not set
        if self.isLimit and self.latestMessageTime + self.minMessageInterval >= time.time():
            print("slackBot ERROR: You're sending message too fast! (%.2f seconds) slackBot.DisableLimit() to disable this feature"%self.minMessageInterval)
            return
        
        # Message convention
        if message == "":
            if prefix == "":
                message_send = ":football: Ping! from %s\n"%self.processInfo
            else:
                message_send = prefix
        else:
            message_send = prefix +  "```" + message + "```"

        # Set channel to default value if empty
        if channel == "":
            channel = self.channel

        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=message_send
            )
        except SlackApiError as e:
            print("slackBot ERROR: Couldn't send message (%s)"%e.response["error"])
        self.latestMessageTime = time.time()

slackBot = slackBot_()
