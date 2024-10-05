import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import earthnet as en

class EarthnetTrainDataset(Dataset):

    def __init__(self, dataDir, dtype=np.float16, transform=None, cropMesodynamic=False):
        
        self.dataDir = Path(dataDir)
        assert self.dataDir.exists(), "Directory to data folder does not exist."
            
        self.dtype = dtype
        self.cubesPathList = sorted(list(self.dataDir.glob("**/*.npz")))
        self.transform = transform
        self.cropMesodynamic = cropMesodynamic
        if cropMesodynamic:
            self.upsample = torch.nn.Upsample(size=(80, 80))

    def __len__(self):
        return len(self.cubesPathList)
    
    def __getitem__(self, index):
        
        cubeFile = np.load(self.cubesPathList[index])
        split    = os.path.split(self.cubesPathList[index])
        tile     = os.path.split(split[0])[1]
        cubename = split[1]

        highresdynamic = cubeFile["highresdynamic"].astype(self.dtype)[:, :, [0, 1, 2, 3], :] # Keep only [blue, green, red, nir] channels HWCT C=4 T=30
        mask           = (1 - cubeFile["highresdynamic"].astype(self.dtype)[:, :, 6, :])[:, :, np.newaxis, :] # Isolate mask channel HWCT C=1 T=30
        highresstatic  = cubeFile["highresstatic"].astype(self.dtype)                         # HWC C=1
        mesodynamic    = cubeFile["mesodynamic"].astype(self.dtype)                           # hwCT C=5 T=150
        mesostatic     = cubeFile["mesostatic"].astype(self.dtype)                            # hwC C=1

        highresdynamic = np.nan_to_num(highresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        highresdynamic = np.clip(highresdynamic, a_min=0.0, a_max=1.0)
        mesodynamic    = np.nan_to_num(mesodynamic, copy=False, nan=0.0)
        highresstatic  = np.nan_to_num(highresstatic, copy=False, nan=0.0)
        mesostatic     = np.nan_to_num(mesostatic, copy=False, nan=0.0)

        if self.cropMesodynamic:
            mesodynamic = mesodynamic[39:41, 39:41, :, :].transpose(2, 3, 0, 1)
            mesodynamic = self.upsample(torch.from_numpy(mesodynamic)).numpy().transpose(2, 3, 0, 1)


        data = {
            "context": {
                "images" : highresdynamic[:, :, :, :10].transpose(2, 3, 0, 1),  # CTHW C=4 T=10(context)
                "weather": mesodynamic[:, :, :, :50].transpose(2, 3, 0, 1),     # CThw C=5 T=50(context)
                "mask"   : mask[:, :, :, :10].transpose(2, 3, 0, 1)             # CTHW C=1 T=10(context)
            },
            "target": {
                "images" : highresdynamic[:, :, :, 10::].transpose(2, 3, 0, 1), # CTHW C=4 T=20(target)
                "weather": mesodynamic[:, :, :, 50::].transpose(2, 3, 0, 1),    # CThw C=5 T=100(target)
                "mask"   : mask[:, :, :, 10::].transpose(2, 3, 0, 1)            # CTHW C=1 T=20(target)
            },
            "demHigh" : highresstatic.transpose(2, 0, 1),                       # CHW C=1
            "demMeso" : mesostatic.transpose(2, 0, 1),                          # Chw C=1
            "tile"    : tile,
            "cubename": cubename
        }

        if self.transform:
            return self.transform(data)
        
        return data


class EarthnetTestDataset(Dataset):

    def __init__(self, dataDir, dtype=np.float16, transform=None, cropMesodynamic=False):
        
        self.dataDir = Path(dataDir)
        assert self.dataDir.exists(), "Directory to data folder does not exist."
        
        subfolders = [f.name for f in self.dataDir.iterdir() if f.is_dir()]
        assert set(['context', 'target']).issubset(set(subfolders)), "Context and/or target subfolders do not exist."

        self.contextPathList = sorted(list(self.dataDir.joinpath('context').glob("**/*.npz")))
        self.targetPathList  = sorted(list(self.dataDir.joinpath('target').glob("**/*.npz")))
    
        self.dtype = dtype
        self.transform = transform

        self.cropMesodynamic = cropMesodynamic
        if cropMesodynamic:
            self.upsample = torch.nn.Upsample(size=(80, 80))

    def __len__(self):
        return len(self.contextPathList)
    
    def __getitem__(self, index):
        contextCubeFile = np.load(self.contextPathList[index])
        targetCubeFile  = np.load(self.targetPathList[index])
        pathString      = str(self.targetPathList[index])
        tile            = pathString.split("/")[-2]
        cubename        = pathString.split("/")[-1]

        contextHighresdynamic = contextCubeFile["highresdynamic"].astype(self.dtype)[:, :, [0, 1, 2, 3], :] # Keep only [blue, green, red, nir] channels HWCT C=4 T=10
        contextMask           = (1 - contextCubeFile["highresdynamic"].astype(self.dtype)[:, :, 4, :])[:, :, np.newaxis, :] # Isolate mask channel HWCT C=1 T=10
        contextHighresstatic  = contextCubeFile["highresstatic"].astype(self.dtype)                         # HWC C=1
        contextMesodynamic    = contextCubeFile["mesodynamic"].astype(self.dtype)                           # hwCT C=5 T=150
        contextMesostatic     = contextCubeFile["mesostatic"].astype(self.dtype)                            # hwC C=1

        contextHighresdynamic = np.nan_to_num(contextHighresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        contextHighresdynamic = np.clip(contextHighresdynamic, a_min=0.0, a_max=1.0)
        contextHighresstatic    = np.nan_to_num(contextHighresstatic, copy=False, nan=0.0)
        contextMesodynamic  = np.nan_to_num(contextMesodynamic, copy=False, nan=0.0)
        contextMesostatic     = np.nan_to_num(contextMesostatic, copy=False, nan=0.0)

        if self.cropMesodynamic:
            contextMesodynamic = contextMesodynamic[39:41, 39:41, :, :].transpose(2, 3, 0, 1)
            contextMesodynamic = self.upsample(torch.from_numpy(contextMesodynamic)).numpy().transpose(2, 3, 0, 1)

        targetHighresdynamic = targetCubeFile["highresdynamic"].astype(self.dtype)[:, :, [0, 1, 2, 3], :] # Keep only [blue, green, red, nir] channels HWCT C=4 T=20
        targetMask           = (1 - targetCubeFile["highresdynamic"].astype(self.dtype)[:, :, 4, :])[:, :, np.newaxis, :] # Isolate mask channel HWCT C=1 T=20

        targetHighresdynamic = np.nan_to_num(targetHighresdynamic, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        targetHighresdynamic = np.clip(targetHighresdynamic, a_min=0.0, a_max=1.0)

        data = {
            "context": {
                "images" : contextHighresdynamic.transpose(2, 3, 0, 1),  # CTHW C=4 T=10(context)
                "weather": contextMesodynamic[:, :, :, :50].transpose(2, 3, 0, 1),     # CThw C=5 T=50(context)
                "mask"   : contextMask.transpose(2, 3, 0, 1)             # CTHW C=1 T=10(context)
            },
            "target": {
                "images" : targetHighresdynamic.transpose(2, 3, 0, 1), # CTHW C=4 T=20(target)
                "weather": contextMesodynamic[:, :, :, 50::].transpose(2, 3, 0, 1),    # CThw C=5 T=100(target)
                "mask"   : targetMask.transpose(2, 3, 0, 1)            # CTHW C=1 T=20(target)
            },
            "demHigh" : contextHighresstatic.transpose(2, 0, 1),                       # CHW C=1
            "demMeso" : contextMesostatic.transpose(2, 0, 1),                          # Chw C=1
            "tile"    : tile,
            "cubename": cubename
        }

        if self.transform:
            return self.transform(data)
        
        return data


class Preprocessing(object):

    def __init__(self):
        None

    def __call__(self, data):

        contextImages = torch.from_numpy(data['context']['images'])
        contextWeather = torch.from_numpy(data['context']['weather'])
        contextMask = torch.from_numpy(data['context']['mask'])
        targetImages = torch.from_numpy(data['target']['images'])
        targetWeather = torch.from_numpy(data['target']['weather'])
        targetMask = torch.from_numpy(data['target']['mask'])
        demHigh = torch.from_numpy(data['demHigh'])
        demMeso = torch.from_numpy(data['demMeso'])

        H, W = contextImages.shape[2::]

        allWeather = torch.cat((contextWeather, targetWeather), 1) # Concatenate all weather data across time to get full 150 days
        allWeather = F.interpolate(allWeather, size=(H, W))         # CThw -> CTHW T=150
        allWeather = allWeather.reshape(allWeather.shape[0], 10, 15, H, W).mean(2).reshape(allWeather.shape[0], 10, H, W) # CTHW T=10
        
        demHigh = demHigh.unsqueeze(1)   # from CHW -> CTHW C=1 T=1
        demHigh = torch.repeat_interleave(demHigh, repeats=10, dim=1)  # CTHW T=10
        

        demMeso = demMeso.unsqueeze(1)                  # from Chw -> CThw C=1 T=1
        demMeso = F.interpolate(demMeso, size=(H, W))   # CTHW C=1 T=1
        demMeso = torch.repeat_interleave(demMeso, repeats=10, dim=1) # CTHW T=10

        data = {
            "x": torch.cat((contextImages, allWeather, demHigh, demMeso)),
            "y": targetImages,
            "targetMask": targetMask,
            "tile"    : data['tile'],
            "cubename": data['cubename']
        }

        return data

# Requires update
class PreprocessingV2(object):

    def __init__(self):
        None

    def __call__(self, data):

        contextImages = torch.from_numpy(data['context']['images'])
        contextWeather = torch.from_numpy(data['context']['weather'])
        contextMask = torch.from_numpy(data['context']['mask'])
        targetImages = torch.from_numpy(data['target']['images'])
        targetWeather = torch.from_numpy(data['target']['weather'])
        targetMask = torch.from_numpy(data['target']['mask'])
        demHigh = torch.from_numpy(data['demHigh'])
        demMeso = torch.from_numpy(data['demMeso'])

        H, W = contextImages.shape[2::]

        contextWeather = F.interpolate(contextWeather, size=(H, W))         # CThw -> CTHW T=50
        contextWeather = contextWeather.reshape(contextWeather.shape[0], 10, 5, H, W).mean(2).reshape(contextWeather.shape[0], 10, H, W) # CTHW T=50 -> CTHW T=10

        demHigh = demHigh.unsqueeze(1)   # from CHW -> CTHW C=1 T=1
        demHigh = torch.repeat_interleave(demHigh, repeats=10, dim=1)  # CTHW T=10
        

        demMeso = demMeso.unsqueeze(1)                  # from Chw -> CThw C=1 T=1
        demMeso = F.interpolate(demMeso, size=(H, W))   # CTHW C=1 T=1
        demMeso = torch.repeat_interleave(demMeso, repeats=10, dim=1) # CTHW T=10

        targetWeather = F.interpolate(targetWeather, size=(H, W))         # CThw -> CTHW T=100
        targetWeather = targetWeather.reshape(targetWeather.shape[0], 20, 5, H, W).mean(2).reshape(targetWeather.shape[0], 20, H, W) # CTHW T=100 -> CTHW T=20



        # X, Y = sample
        
        # X["highresdynamic"] = torch.permute(torch.from_numpy(X["highresdynamic"]).unsqueeze(0), (0, 3, 4, 1, 2))    # X["highresdynamic"] from HWCT -> BHWCT -> BCTHW

        # X["highresstatic"] = torch.unsqueeze(torch.from_numpy(X["highresstatic"]).unsqueeze(0), -1)                 # X["highresstatic"] from HWC -> BHWC -> BHWCT T=1
        # X["highresstatic"] = torch.permute(X["highresstatic"], (0, 3, 4, 1, 2))                                     # BHWCT -> BCTHW (T=1)
        # X["highresstatic"] = F.interpolate(X["highresstatic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (T=10)

        # X["mesodynamic"] = torch.permute(torch.from_numpy(X["mesodynamic"]).unsqueeze(0), (0, 3, 4, 1, 2))          # X["mesodynamic"] from hwCT -> BhwCT -> BCThw (T=150)
        # xMesodynamicContext = X["mesodynamic"][:, :, :50, :, :]
        # xMesodynamicTarget = X["mesodynamic"][:, :, 50::, :, :]
        # xMesodynamicContext = F.interpolate(xMesodynamicContext, (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (from T=50 to T=10)
        # xMesodynamicTarget = F.interpolate(xMesodynamicTarget, (20, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (from T=100 to T=20)
        # xMesodynamicTarget = xMesodynamicTarget.squeeze(0)                                                                       #BCTHW -> CTHW
        # # X["mesodynamic"] = F.interpolate(X["mesodynamic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (T=10)

        # X["mesostatic"] = torch.unsqueeze(torch.from_numpy(X["mesostatic"]).unsqueeze(0), -1)                       # X["mesostatic"] from hwC -> BhwC -> BhwCT T=1
        # X["mesostatic"] = torch.permute(X["mesostatic"], (0, 3, 4, 1, 2))                                           # BhwCT -> BCThw (T=1)
        # X["mesostatic"] = F.interpolate(X["mesostatic"], (10, X["highresdynamic"].shape[3], X["highresdynamic"].shape[4])) # BCTHW (T=10)

        # x = torch.cat((X["highresdynamic"], X["highresstatic"], xMesodynamicContext, X["mesostatic"]), 1).squeeze(0)   # BCTHW concat by C -> CTHW

        # y = torch.permute(torch.from_numpy(Y["highresdynamic"]).unsqueeze(0), ((0, 3, 4, 1, 2))).squeeze(0)         # Y["highresdynamic"] from HWCT -> BHWCT -> BCTHW -> CTHW

        data = {
            "xContext": torch.cat((contextImages, contextWeather, demHigh, demMeso)),
            "xTarget" : targetWeather,
            "y": targetImages,
            "targetMask": targetMask,
            "tile"    : data['tile'],
            "cubename": data['cubename'],
        }

        return data
    

class PreprocessingStack(object):

    def __init__(self):
        None

    def __call__(self, data):

        contextImages = torch.from_numpy(data['context']['images'])
        contextWeather = torch.from_numpy(data['context']['weather'])
        contextMask = torch.from_numpy(data['context']['mask'])
        targetImages = torch.from_numpy(data['target']['images'])
        targetWeather = torch.from_numpy(data['target']['weather'])
        targetMask = torch.from_numpy(data['target']['mask'])
        demHigh = torch.from_numpy(data['demHigh'])
        demMeso = torch.from_numpy(data['demMeso'])

        H, W = contextImages.shape[2::]

        allWeather = torch.cat((contextWeather, targetWeather), 1) # Concatenate all weather data across time to get full 150 days
        allWeather = F.interpolate(allWeather, size=(H, W))         # CThw -> CTHW T=150
        allWeather = allWeather.reshape(allWeather.shape[0], 10, 15, H, W).permute(0, 2, 1, 3, 4).reshape(allWeather.shape[0]*15, 10, H, W) # CTHW C=5*15 T=10
        
        demHigh = demHigh.unsqueeze(1)   # from CHW -> CTHW C=1 T=1
        demHigh = torch.repeat_interleave(demHigh, repeats=10, dim=1)  # CTHW T=10
        

        demMeso = demMeso.unsqueeze(1)                  # from Chw -> CThw C=1 T=1
        demMeso = F.interpolate(demMeso, size=(H, W))   # CTHW C=1 T=1
        demMeso = torch.repeat_interleave(demMeso, repeats=10, dim=1) # CTHW T=10

        data = {
            "x": torch.cat((contextImages, allWeather, demHigh, demMeso)), # C=81
            "y": targetImages,
            "targetMask": targetMask,
            "tile"    : data['tile'],
            "cubename": data['cubename']
        }

        return data
    
class PreprocessingSeparate(object):

    def __init__(self):
        None

    def __call__(self, data):

        contextImages = torch.from_numpy(data['context']['images'])
        contextWeather = torch.from_numpy(data['context']['weather'])
        contextMask = torch.from_numpy(data['context']['mask'])
        targetImages = torch.from_numpy(data['target']['images'])
        targetWeather = torch.from_numpy(data['target']['weather'])
        targetMask = torch.from_numpy(data['target']['mask'])
        demHigh = torch.from_numpy(data['demHigh'])
        demMeso = torch.from_numpy(data['demMeso'])

        H, W = contextImages.shape[2::]

        h, w = contextWeather.shape[2::]

        # contextWeather = F.interpolate(contextWeather, size=(H, W))         # CThw -> CTHW T=50
        contextWeather = contextWeather.reshape(contextWeather.shape[0], 10, 5, h, w).mean(2).reshape(contextWeather.shape[0], 10, h, w) # CTHW T=50 -> CTHW T=10

        # demHigh = demHigh.unsqueeze(1)   # from CHW -> CTHW C=1 T=1
        # demHigh = torch.repeat_interleave(demHigh, repeats=10, dim=1)  # CTHW T=10
        

        # demMeso = demMeso.unsqueeze(1)                  # from Chw -> CThw C=1 T=1
        # demMeso = F.interpolate(demMeso, size=(H, W))   # CTHW C=1 T=1
        # demMeso = torch.repeat_interleave(demMeso, repeats=10, dim=1) # CTHW T=10

        # targetWeather = F.interpolate(targetWeather, size=(H, W))         # CThw -> CTHW T=100
        targetWeather = targetWeather.reshape(targetWeather.shape[0], 20, 5, h, w).mean(2).reshape(targetWeather.shape[0], 20, h, w) # CTHW T=100 -> CTHW T=20

        data = {
            "contextImg": contextImages,
            "contextWeather": contextWeather,
            "targetWeather": targetWeather,
            "staticData": demHigh,
            "y": targetImages,
            "targetMask": targetMask,
            "tile"    : data['tile'],
            "cubename": data['cubename'],
        }

        return data
    
class PreprocessingWeather(object):

    def __init__(self, reduceTime):
        self.reduceTime = reduceTime

    def __call__(self, data):

        contextWeather = torch.from_numpy(data['context']['weather'])
        targetWeather = torch.from_numpy(data['target']['weather'])

        h, w = contextWeather.shape[2::]

        if self.reduceTime:
            # contextWeather = F.interpolate(contextWeather, size=(H, W))         # CThw -> CTHW T=50
            contextWeather = contextWeather.reshape(contextWeather.shape[0], 10, 5, h, w).mean(2).reshape(contextWeather.shape[0], 10, h, w) # CThw T=50 -> CThw T=10

        # demHigh = demHigh.unsqueeze(1)   # from CHW -> CTHW C=1 T=1
        # demHigh = torch.repeat_interleave(demHigh, repeats=10, dim=1)  # CTHW T=10
        

        # demMeso = demMeso.unsqueeze(1)                  # from Chw -> CThw C=1 T=1
        # demMeso = F.interpolate(demMeso, size=(H, W))   # CTHW C=1 T=1
        # demMeso = torch.repeat_interleave(demMeso, repeats=10, dim=1) # CTHW T=10

            # targetWeather = F.interpolate(targetWeather, size=(H, W))         # CThw -> CTHW T=100
            targetWeather = targetWeather.reshape(targetWeather.shape[0], 20, 5, h, w).mean(2).reshape(targetWeather.shape[0], 20, h, w) # CThw T=100 -> CThw T=20

        data = {
            "contextWeather": contextWeather,
            "targetWeather": targetWeather,
            "tile"    : data['tile'],
            "cubename": data['cubename'],
        }

        return data
    
class PreprocessingV7(object):

    def __init__(self, reduceTime):
        self.reduceTime = reduceTime

    def __call__(self, data):

        contextImages = torch.from_numpy(data['context']['images'])
        contextWeather = torch.from_numpy(data['context']['weather'])
        contextMask = torch.from_numpy(data['context']['mask'])
        targetImages = torch.from_numpy(data['target']['images'])
        targetWeather = torch.from_numpy(data['target']['weather'])
        targetMask = torch.from_numpy(data['target']['mask'])
        demHigh = torch.from_numpy(data['demHigh'])
        demMeso = torch.from_numpy(data['demMeso'])

        H, W = contextImages.shape[2::]

        h, w = contextWeather.shape[2::]

        if self.reduceTime:
            # contextWeather = F.interpolate(contextWeather, size=(H, W))         # CThw -> CTHW T=50
            contextWeather = contextWeather.reshape(contextWeather.shape[0], 10, 5, h, w).mean(2).reshape(contextWeather.shape[0], 10, h, w) # CThw T=50 -> CThw T=10
            # targetWeather = F.interpolate(targetWeather, size=(H, W))         # CThw -> CTHW T=100
            targetWeather = targetWeather.reshape(targetWeather.shape[0], 20, 5, h, w).mean(2).reshape(targetWeather.shape[0], 20, h, w) # CThw T=100 -> CThw T=20

        demHigh = demHigh.unsqueeze(1)   # from CHW -> CTHW C=1 T=1
        demHigh = torch.repeat_interleave(demHigh, repeats=10, dim=1)  # CTHW T=10
        
        # demMeso = demMeso.unsqueeze(1)                  # from Chw -> CThw C=1 T=1
        # demMeso = F.interpolate(demMeso, size=(H, W))   # CTHW C=1 T=1
        # demMeso = torch.repeat_interleave(demMeso, repeats=10, dim=1) # CTHW T=10

        data = {
            "contextImgDEM": torch.cat((contextImages, demHigh)), # CTHW, C=5, T=10
            "contextWeather": contextWeather,                     # CThw, C=5, T=10
            "targetWeather": targetWeather,                       # CThw, C=5, T=20
            "y": targetImages,                                    # CTHW, C=4, T=20
            "targetMask": targetMask,                             # CTHW, C=1, T=20
            "tile"    : data['tile'],
            "cubename": data['cubename'],
        }

        return data

class PreprocessingV8(object):

    def __init__(self):
        pass

    def __call__(self, data):

        contextImages = torch.from_numpy(data['context']['images'])
        contextWeather = torch.from_numpy(data['context']['weather'])
        contextMask = torch.from_numpy(data['context']['mask'])
        targetImages = torch.from_numpy(data['target']['images'])
        targetWeather = torch.from_numpy(data['target']['weather'])
        targetMask = torch.from_numpy(data['target']['mask'])
        demHigh = torch.from_numpy(data['demHigh'])
        demMeso = torch.from_numpy(data['demMeso'])

        H, W = contextImages.shape[2::]

        h, w = contextWeather.shape[2::]

        allWeather = torch.cat((contextWeather, targetWeather), 1) # Concatenate all weather data across time to get full 150 days
        allWeather = F.interpolate(allWeather, size=(H, W))         # CThw -> CTHW T=150
        allWeather = allWeather.reshape(allWeather.shape[0], 10, 15, H, W).mean(2).reshape(allWeather.shape[0], 10, H, W) # CTHW T=10

        # demHigh = demHigh.unsqueeze(1)   # from CHW -> CTHW C=1 T=1
        # demHigh = torch.repeat_interleave(demHigh, repeats=10, dim=1)  # CTHW T=10
        
        # demMeso = demMeso.unsqueeze(1)                  # from Chw -> CThw C=1 T=1
        # demMeso = F.interpolate(demMeso, size=(H, W))   # CTHW C=1 T=1
        # demMeso = torch.repeat_interleave(demMeso, repeats=10, dim=1) # CTHW T=10

        data = {
            "contextImgWeather": torch.cat((contextImages, allWeather)), # CTHW, C=9, T=10
            "demHigh": demHigh,                                          # CHW, C=1
            "y": targetImages,                                           # CTHW, C=4, T=20
            "targetMask": targetMask,                                    # CTHW, C=1, T=20
            "tile"    : data['tile'],
            "cubename": data['cubename'],
        }

        return data
    
    
if __name__ == "__main__":

    # Set paremeters
    BATCH_SIZE           = 1 # Bactch size
    NUM_WORKERS          = 2  # Number of workers for Dataloaders

    v = '2'

    if v == '1':
        preprocessingStage = Preprocessing()
    elif v == '2':
        preprocessingStage = PreprocessingV2()
    else:
        preprocessingStage = None

    # Create dataset of training part of Earthnet dataset
    trainDataset = EarthnetTrainDataset(dataDir='/home/nikoskot/earthnetThesis/EarthnetDataset/train', dtype=np.float32, transform=preprocessingStage)

    # Create training and validation Dataloaders
    trainDataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Create dataset of training part of Earthnet dataset
    testDataset = EarthnetTestDataset(dataDir='/home/nikoskot/earthnetThesis/EarthnetDataset/iid_test_split', dtype=np.float32, transform=preprocessingStage)

    # Create training and validation Dataloaders
    testDataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    if preprocessingStage == None:
        print("Training dataloader:")
        print("Length {}".format(len(trainDataloader)))
        data = next(iter(trainDataloader))
        print(data['context']['images'].shape)
        print(data['context']['weather'].shape)
        print(data['context']['mask'].shape)
        print(data['target']['images'].shape)
        print(data['target']['weather'].shape)
        print(data['target']['mask'].shape)
        print(data['demHigh'].shape)
        print(data['demMeso'].shape)
        print(data['tile'])
        print(data['cubename'])

        print("Testing dataloader sample")
        print("Length {}".format(len(testDataloader)))
        data = next(iter(testDataloader))
        print(data['context']['images'].shape)
        print(data['context']['weather'].shape)
        print(data['context']['mask'].shape)
        print(data['target']['images'].shape)
        print(data['target']['weather'].shape)
        print(data['target']['mask'].shape)
        print(data['demHigh'].shape)
        print(data['demMeso'].shape)
        print(data['tile'])
        print(data['cubename'])
    elif v == '1':
        print("Training dataloader:")
        print("Length {}".format(len(trainDataloader)))
        data = next(iter(trainDataloader))
        print(data['x'].shape)
        print(data['y'].shape)
        print(data['targetMask'].shape)
        print(data['tile'])
        print(data['cubename'])
        print(torch.max(data['x']))
        print(torch.min(data['x']))
        print(torch.max(data['y']))
        print(torch.min(data['y']))
        print(torch.unique(data['targetMask']))

        np.savez_compressed('/home/nikoskot/6', highresdynamic=data['y'][0].permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.float16))
        en.cube_gallery('/home/nikoskot/6.npz', variable='rgb', save_path='/home/nikoskot/6rgb')

        print("Testing dataloader sample")
        print("Length {}".format(len(testDataloader)))
        data = next(iter(testDataloader))
        print(data['x'].shape)
        print(data['y'].shape)
        print(data['targetMask'].shape)
        print(data['tile'])
        print(data['cubename'])
    elif v == '2':
        print("Training dataloader:")
        print("Length {}".format(len(trainDataloader)))
        data = next(iter(trainDataloader))
        print(data['xContext'].shape)
        print(data['xTarget'].shape)
        print(data['y'].shape)
        print(data['targetMask'].shape)
        print(data['tile'])
        print(data['cubename'])
        print(torch.max(data['xContext']))
        print(torch.min(data['xContext']))
        print(torch.max(data['xTarget']))
        print(torch.min(data['xTarget']))
        print(torch.max(data['y']))
        print(torch.min(data['y']))
        print(torch.unique(data['targetMask']))

        np.savez_compressed('/home/nikoskot/7', highresdynamic=data['y'][0].permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.float16))
        en.cube_gallery('/home/nikoskot/7.npz', variable='rgb', save_path='/home/nikoskot/7rgb')

        print("Testing dataloader sample")
        print("Length {}".format(len(testDataloader)))
        data = next(iter(testDataloader))
        print(data['xContext'].shape)
        print(data['xTarget'].shape)
        print(data['y'].shape)
        print(data['targetMask'].shape)
        print(data['tile'])
        print(data['cubename'])

    # Explore dataset

