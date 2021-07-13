class Stat():
    def __init__(self, args):
        self.loss = []
        self.testAccuracy = []
        self.trainAccuracy = []
        self.running_loss = 0.0
        self.loss_count = 0
        self.loss_batches = args.stat_loss_batches
        self.file_location = args.stat_location

    def AddLoss(self, v):
        self.running_loss += v
        self.loss_count += 1
        if self.loss_count == self.loss_batches:
            self.loss.append(self.running_loss / self.loss_batches)
            self.loss_count = 0
            self.running_loss = 0.0
    
    def AddTestAccuracy(self, v):
        self.testAccuracy.append(v)

    def AddTrainAccuracy(self, v):
        self.trainAccuracy.append(v)

    def SaveToFile(self):
        if self.loss_count != 0:
            self.loss.append(self.running_loss / self.loss_batches)
        
        f = open(self.file_location, mode="w+", newline='', encoding='utf-8')
        f.write(">Average Loss per {} batches\n".format(self.loss_batches))
        for i in self.loss:
            f.write(str(i)+"\t")
        f.write("\n")
        f.write("> Test Accuracy\n")
        for i in self.testAccuracy:
            f.write(str(i)+"\t")
        f.write("\n")
        if len(self.trainAccuracy) > 0:
            f.write("> Train Accuracy\n")
            for i in self.trainAccuracy:
                f.write(str(i)+"\t")
            f.write("\n")

class statManager_:
    def __init__(self):
        pass

statManager = statManager_