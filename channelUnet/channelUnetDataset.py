from typing import Union
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class EarthNet2021Dataset(Dataset):

    def __init__(self, folder: Union[Path, str], noisy_masked_pixels = False, use_meso_static_as_dynamic = False, fp16 = False):
        if not isinstance(folder, Path):
            folder = Path(folder)
        assert (not {"target","context"}.issubset(set([d.name for d in folder.glob("*") if d.is_dir()])))

        self.filepaths = sorted(list(folder.glob("**/*.npz")))

        self.noisy_masked_pixels = noisy_masked_pixels
        self.use_meso_static_as_dynamic = use_meso_static_as_dynamic
        self.type = np.float16 if fp16 else np.float32

    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]

        npz = np.load(filepath)

        highresdynamic = np.transpose(npz["highresdynamic"],(3,2,0,1)).astype(self.type)
        highresstatic = np.transpose(npz["highresstatic"],(2,0,1)).astype(self.type)
        mesodynamic = np.transpose(npz["mesodynamic"],(3,2,0,1)).astype(self.type)
        mesostatic = np.transpose(npz["mesostatic"],(2,0,1)).astype(self.type)

        masks = ((1 - highresdynamic[:,-1,:,:])[:,np.newaxis,:,:]).repeat(4,1)

        images = highresdynamic[:,:4,:,:]
        
        images[np.isnan(images)] = 0
        images[images > 1] = 1
        images[images < 0] = 0
        mesodynamic[np.isnan(mesodynamic)] = 0
        highresstatic[np.isnan(highresstatic)] = 0
        mesostatic[np.isnan(mesostatic)] = 0

        if self.noisy_masked_pixels:            
            images = np.transpose(images,(1,0,2,3))
            all_pixels = images[np.transpose(masks, (1,0,2,3)) == 1].reshape(4,-1)
            all_pixels = np.stack(int(images.size/all_pixels.size+1)*[all_pixels],axis = 1)
            all_pixels = all_pixels.reshape(4,-1)
            all_pixels = all_pixels.transpose(1,0)
            np.random.shuffle(all_pixels)
            all_pixels = all_pixels.transpose(1,0)
            all_pixels = all_pixels[:,:images.size//4].reshape(*images.shape)
            images = np.where(np.transpose(masks, (1,0,2,3)) == 0, all_pixels, images)
            images = np.transpose(images,(1,0,2,3))

        if self.use_meso_static_as_dynamic:
            mesodynamic = np.concatenate([mesodynamic, mesostatic[np.newaxis, :, :, :].repeat(mesodynamic.shape[0], 0)], axis = 1)

        data = {
            "dynamic": [
                torch.from_numpy(images),
                torch.from_numpy(mesodynamic)
            ],
            "dynamic_mask": [
                torch.from_numpy(masks)
            ],
            "static": [
                torch.from_numpy(highresstatic),
                torch.from_numpy(mesostatic)
            ] if not self.use_meso_static_as_dynamic else [
                torch.from_numpy(highresstatic)
            ],
            "static_mask": [],
            "filepath": str(filepath),
            "cubename": self.__name_getter(filepath)
        }

        return data

    def __len__(self) -> int:
        return len(self.filepaths)

    def __name_getter(self, path: Path) -> str:
        """Helper function gets Cubename from a Path

        Args:
            path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

        Returns:
            [str]: cubename (has format tile_startyear_startmonth_startday_endyear_endmonth_endday_hrxmin_hrxmax_hrymin_hrymax_mesoxmin_mesoxmax_mesoymin_mesoymax.npz)
        """        
        components = path.name.split("_")
        regex = re.compile('\d{2}[A-Z]{3}')
        if bool(regex.match(components[0])):
            return path.name
        else:
            assert(bool(regex.match(components[1])))
            return "_".join(components[1:]) 