import torch

from utils.logger import Log

def SaveModel(args, suffix):
    PATH = "%s/%s.model"%(args.save_prefix,suffix)
    Log.Print("Saving model file as %s"%PATH)
    torch.save(args.net.state_dict(), PATH)

def SaveState(args, state = True, model = True, suffix = ""):
    if model:
        if suffix == "":
            save_path_model = "%s.model"%(args.save_prefix)
        else:
            save_path_model = "%s_%s.model"%(args.save_prefix, suffix)
        torch.save(args.net.state_dict(), save_path_model)
        Log.Print("Saved model to %s"%save_path_model)
    if state:
        if suffix == "":
            save_path_state = "%s.model"%(args.save_prefix)
        else:
            save_path_state = "%s_%s.state"%(args.save_prefix, suffix)
        torch.save({
            'model_state_dict':args.net.state_dict(),
            'optimizer_state_dict':args.optimizer.state_dict()
        }, save_path_state)
        Log.Print("Saved state to %s"%save_path_state)

def LoadModel(args, path):
    args.load_state_dict(torch.load(path))
    args.net.eval()
    Log.Print("Loaded model from %s"%path)


def LoadState(args, path):
    checkpoint = torch.load(path)
    
    args.net.load_state_dict(checkpoint['model_state_dict'])
    args.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    Log.Print("Loaded state from %s"%path)