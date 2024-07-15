import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

class EarthnetTrainDataset(Dataset):

    def __init__(self, dataDir, dtype=np.float16, transform=None):
        
        self.dataDir = Path(dataDir)
        assert self.dataDir.exists(), "Directory to data folder does not exist."
            
        self.dtype = dtype
        self.cubesPathList = sorted(list(self.dataDir.glob("**/*.npz")))
        self.transform = transform

    def __len__(self):
        return len(self.cubesPathList)
    
    def __getitem__(self, index):
        
        cubeFile = np.load(self.cubesPathList[index])
        split    = os.path.split(self.cubesPathList[index])
        tile     = os.path.split(split[0])[1]
        cubename = split[1]

        # keep only [blue, green, red, nir, mask] channels
        highresdynamic = cubeFile["highresdynamic"].astype(self.dtype)[:, :, [0, 1, 2, 3, 6], :]
        highresstatic  = cubeFile["highresstatic"].astype(self.dtype)
        mesodynamic    = cubeFile["mesodynamic"].astype(self.dtype)
        mesostatic     = cubeFile["mesostatic"].astype(self.dtype)

        highresdynamic = np.nan_to_num(highresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        highresdynamic = np.clip(highresdynamic, a_min=0.0, a_max=1.0)
        mesodynamic    = np.nan_to_num(mesodynamic, copy=False, nan=0.0)
        highresstatic  = np.nan_to_num(highresstatic, copy=False, nan=0.0)
        mesostatic     = np.nan_to_num(mesostatic, copy=False, nan=0.0)

        X = {
            "highresdynamic": highresdynamic[..., :10], # HWCT C=5 T=10(context)
            "highresstatic" : highresstatic,            # HWC  C=1
            "mesodynamic"   : mesodynamic,              # hwCT C=5 T=150
            "mesostatic"    : mesostatic,               # hwC  C=1
            }
        
        Y = {
            "highresdynamic": highresdynamic[..., 10::] # HWCT C=5 T=20(target) contains mask
            }
        
        xMesodynamicTarget = None
                
        if self.transform:
            X, Y, xMesodynamicTarget = self.transform((X, Y))
        
        return X, Y, tile, cubename,  xMesodynamicTarget    # CTHW, CTHW


class EarthnetTestDataset(Dataset):

    def __init__(self, dataDir, dtype=np.float16, transform=None):
        
        self.dataDir = Path(dataDir)
        assert self.dataDir.exists(), "Directory to data folder does not exist."
        
        subfolders = [f.name for f in self.dataDir.iterdir() if f.is_dir()]
        assert set(['context', 'target']).issubset(set(subfolders)), "Context and/or target subfolders do not exist."

        self.contextPathList = sorted(list(self.dataDir.joinpath('context').glob("**/*.npz")))
        self.targetPathList  = sorted(list(self.dataDir.joinpath('target').glob("**/*.npz")))
    
        self.dtype = dtype
        self.transform = transform

    def __len__(self):
        return len(self.contextPathList)
    
    def __getitem__(self, index):
        contextCubeFile = np.load(self.contextPathList[index])
        targetCubeFile  = np.load(self.targetPathList[index])
        pathString      = str(self.targetPathList[index])
        tile            = pathString.split("/")[-2]
        cubename        = pathString.split("/")[-1]

        # keep only [blue, green, red, nir, mask] channels
        contextHighresdynamic = contextCubeFile["highresdynamic"].astype(self.dtype)
        contextHighresstatic  = contextCubeFile["highresstatic"].astype(self.dtype)
        contextMesodynamic    = contextCubeFile["mesodynamic"].astype(self.dtype)
        contextMesostatic     = contextCubeFile["mesostatic"].astype(self.dtype)

        contextHighresdynamic = np.nan_to_num(contextHighresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        contextHighresdynamic = np.clip(contextHighresdynamic, a_min=0.0, a_max=1.0)
        contextHighresstatic    = np.nan_to_num(contextHighresstatic, copy=False, nan=0.0)
        contextMesodynamic  = np.nan_to_num(contextMesodynamic, copy=False, nan=0.0)
        contextMesostatic     = np.nan_to_num(contextMesostatic, copy=False, nan=0.0)

        # keep only [blue, green, red, nir, mask] channels
        targetHighresdynamic = targetCubeFile["highresdynamic"].astype(self.dtype)
        targetHighresdynamic = np.nan_to_num(targetHighresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        targetHighresdynamic = np.clip(targetHighresdynamic, a_min=0.0, a_max=1.0)

        X = {
            "highresdynamic": contextHighresdynamic,    # HWCT C=5 T=10(context)
            "highresstatic" : contextHighresstatic,     # HWC  C=1
            "mesodynamic"   : contextMesodynamic,       # hwCT C=5 T=150
            "mesostatic"    : contextMesostatic,        # hwC  C=1
            }
        
        Y = {
            "highresdynamic": targetHighresdynamic      # HWCT C=5 T=20(target) contains mask
            }
        
        xMesodynamicTarget = None
        
        if self.transform:
            X, Y, xMesodynamicTarget = self.transform((X, Y))

        return X, Y, tile, cubename, xMesodynamicTarget    # CTHW, CTHW

class Preprocessing(object):

    def __init__(self):
        None

    def __call__(self, sample):

        X, Y = sample
        
        X["highresdynamic"] = torch.permute(torch.from_numpy(X["highresdynamic"]).unsqueeze(0), (0, 3, 4, 1, 2))    # X["highresdynamic"] from HWCT -> BHWCT -> BCTHW

        X["highresstatic"] = torch.unsqueeze(torch.from_numpy(X["highresstatic"]).unsqueeze(0), -1)                 # X["highresstatic"] from HWC -> BHWC -> BHWCT T=1
        X["highresstatic"] = torch.permute(X["highresstatic"], (0, 3, 4, 1, 2))                                     # BHWCT -> BCTHW (T=1)
        X["highresstatic"] = F.interpolate(X["highresstatic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (T=10)

        X["mesodynamic"] = torch.permute(torch.from_numpy(X["mesodynamic"]).unsqueeze(0), (0, 3, 4, 1, 2))          # X["mesodynamic"] from hwCT -> BhwCT -> BCThw (T=150)
        X["mesodynamic"] = F.interpolate(X["mesodynamic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (T=10)

        X["mesostatic"] = torch.unsqueeze(torch.from_numpy(X["mesostatic"]).unsqueeze(0), -1)                       # X["mesostatic"] from hwC -> BhwC -> BhwCT T=1
        X["mesostatic"] = torch.permute(X["mesostatic"], (0, 3, 4, 1, 2))                                           # BhwCT -> BCThw (T=1)
        X["mesostatic"] = F.interpolate(X["mesostatic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (T=10)

        x = torch.cat((X["highresdynamic"], X["highresstatic"], X["mesodynamic"], X["mesostatic"]), 1).squeeze(0)   # BCTHW concat by C -> CTHW

        y = torch.permute(torch.from_numpy(Y["highresdynamic"]).unsqueeze(0), ((0, 3, 4, 1, 2))).squeeze(0)         # Y["highresdynamic"] from HWCT -> BHWCT -> BCTHW -> CTHW

        return x, y, None

class PreprocessingV2(object):

    def __init__(self):
        None

    def __call__(self, sample):

        X, Y = sample
        
        X["highresdynamic"] = torch.permute(torch.from_numpy(X["highresdynamic"]).unsqueeze(0), (0, 3, 4, 1, 2))    # X["highresdynamic"] from HWCT -> BHWCT -> BCTHW

        X["highresstatic"] = torch.unsqueeze(torch.from_numpy(X["highresstatic"]).unsqueeze(0), -1)                 # X["highresstatic"] from HWC -> BHWC -> BHWCT T=1
        X["highresstatic"] = torch.permute(X["highresstatic"], (0, 3, 4, 1, 2))                                     # BHWCT -> BCTHW (T=1)
        X["highresstatic"] = F.interpolate(X["highresstatic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (T=10)

        X["mesodynamic"] = torch.permute(torch.from_numpy(X["mesodynamic"]).unsqueeze(0), (0, 3, 4, 1, 2))          # X["mesodynamic"] from hwCT -> BhwCT -> BCThw (T=150)
        xMesodynamicContext = X["mesodynamic"][:, :, :50, :, :]
        xMesodynamicTarget = X["mesodynamic"][:, :, 50::, :, :]
        xMesodynamicContext = F.interpolate(xMesodynamicContext, (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (from T=50 to T=10)
        xMesodynamicTarget = F.interpolate(xMesodynamicTarget, (20, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (from T=100 to T=20)
        xMesodynamicTarget = xMesodynamicTarget.squeeze(0)                                                                       #BCTHW -> CTHW
        # X["mesodynamic"] = F.interpolate(X["mesodynamic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (T=10)

        X["mesostatic"] = torch.unsqueeze(torch.from_numpy(X["mesostatic"]).unsqueeze(0), -1)                       # X["mesostatic"] from hwC -> BhwC -> BhwCT T=1
        X["mesostatic"] = torch.permute(X["mesostatic"], (0, 3, 4, 1, 2))                                           # BhwCT -> BCThw (T=1)
        X["mesostatic"] = F.interpolate(X["mesostatic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (T=10)

        x = torch.cat((X["highresdynamic"], X["highresstatic"], xMesodynamicContext, X["mesostatic"]), 1).squeeze(0)   # BCTHW concat by C -> CTHW

        y = torch.permute(torch.from_numpy(Y["highresdynamic"]).unsqueeze(0), ((0, 3, 4, 1, 2))).squeeze(0)         # Y["highresdynamic"] from HWCT -> BHWCT -> BCTHW -> CTHW

        return x, y, xMesodynamicTarget
    

if __name__ == "__main__":

    # Set paremeters
    BATCH_SIZE           = 1 # Bactch size
    NUM_WORKERS          = 2  # Number of workers for Dataloaders

    preprocessingStage = PreprocessingV2()
    # preprocessingStage = None

    # Create dataset of training part of Earthnet dataset
    trainDataset = EarthnetTrainDataset(dataDir='/home/nikoskot/EarthnetDataset/train', dtype=np.float32, transform=preprocessingStage)

    # Create training and validation Dataloaders
    trainDataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Create dataset of training part of Earthnet dataset
    testDataset = EarthnetTestDataset(dataDir='/home/nikoskot/EarthnetDataset/iid_test_split', dtype=np.float32, transform=preprocessingStage)

    # Create training and validation Dataloaders
    testDataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    if preprocessingStage == None:
        print("Training dataloader:")
        print("Length {}".format(len(trainDataloader)))
        x, y, t, c, _ = next(iter(trainDataloader))
        print(x['highresdynamic'].shape)
        print(x['highresstatic'].shape)
        print(x['mesodynamic'].shape)
        print(x['mesostatic'].shape)
        print(y['highresdynamic'].shape)
        print(t)
        print(c)

        print("Testing dataloader sample")
        print("Length {}".format(len(testDataloader)))
        x, y, t, c, _ = next(iter(testDataloader))
        print(x['highresdynamic'].shape)
        print(x['highresstatic'].shape)
        print(x['mesodynamic'].shape)
        print(x['mesostatic'].shape)
        print(y['highresdynamic'].shape)
        print(t)
        print(c)
    else:
        print("Training dataloader:")
        print("Length {}".format(len(trainDataloader)))
        x, y, t, c, xMesodynamic = next(iter(trainDataloader))
        print(x.shape)
        print(xMesodynamic.shape)
        print(y.shape)
        print(t)
        print(c)

        print("Testing dataloader sample")
        print("Length {}".format(len(testDataloader)))
        x, y, t, c, xMesodynamic = next(iter(testDataloader))
        print(x.shape)
        print(xMesodynamic.shape)
        print(y.shape)
        print(t)
        print(c)