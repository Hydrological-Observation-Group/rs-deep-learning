''' 
author: xin luo
create: 2025.11.21
des: dataset and dataloader for deep learning tasks in remote sensing
'''
import random
import torch
import rasterio as rio
import numpy as np


## create related functions
## - crop scene to patches
class RandomCrop:
    '''
    des: randomly crop corresponding to specific patch size
    '''
    def __init__(self, size=(256, 256)):
        self.size = size
    def __call__(self, image, truth):
        '''size: (height, width)'''
        start_h = random.randint(0, truth.shape[0]-self.size[0])
        start_w = random.randint(0, truth.shape[1]-self.size[1])
        patch = image[:,start_h:start_h+self.size[0],start_w:start_w+self.size[1]]
        truth = truth[start_h:start_h+self.size[0], start_w:start_w+self.size[1]]
        return patch, truth

### - Dataset definition
class ScenePathSet(torch.utils.data.Dataset):
    '''
    des: build dataset using file paths
    '''
    def __init__(self, paths_scene, paths_truth, path_size=(512, 512)):
        self.paths_scene = paths_scene
        self.paths_truth = paths_truth
        self.path_size = path_size
    def __getitem__(self, idx):
        # Load scene and truth image
        scene_path = self.paths_scene[idx]
        truth_path = self.paths_truth[idx]
        ## 1. read scene and truth images
        with rio.open(scene_path) as src:
            scene_arr = src.read().transpose((1, 2, 0))  # (H, W, C)
        with rio.open(truth_path) as truth_src:
            truth_arr = truth_src.read(1)  # (H, W)
        ## post processing
        scene_arr = scene_arr.astype(np.float32).transpose((2, 0, 1)) # (C, H, W)
        patch, truth = RandomCrop(size=self.path_size)(scene_arr, truth_arr)  # crop
        truth = truth[np.newaxis, :].astype(np.int8)  # (C, H, W)
        patch = torch.from_numpy(patch).float()
        truth = torch.from_numpy(truth).float()
        return patch, truth
    def __len__(self):
        return len(self.paths_scene)

### - Dataset definition
class SceneArraySet(torch.utils.data.Dataset):
    def __init__(self, scenes_arr, truths_arr, path_size=(512, 512)):
        '''
        des: build dataset using pre-loaded arrays
        args:
            scenes_arr: list of np.arrays, each array is (H, W, C) 
            truths_arr: list of np.arrays, each array is (H, W)
            path_size: tuple, (height, width) of cropped patch
        '''
        self.scenes_arr = scenes_arr
        self.truths_arr = truths_arr
        self.path_size = path_size
    def __getitem__(self, idx):
        scene_arr = self.scenes_arr[idx].astype(np.float32).transpose((2, 0, 1)) # (C, H, W)
        truth_arr = self.truths_arr[idx]
        patch, truth = RandomCrop(size=self.path_size)(scene_arr, truth_arr)  # crop
        truth = truth[np.newaxis, :].astype(np.int8)  # (C, H, W)
        patch = torch.from_numpy(patch).float()
        truth = torch.from_numpy(truth).float()
        return patch, truth
    def __len__(self):
        return len(self.scenes_arr)

### - Dataset definition
class PatchSet(torch.utils.data.Dataset):
    def __init__(self, paths_valset):
        self.paths_valset = paths_valset
    def __getitem__(self, idx):
        ## load valset patch, patch: (H, W, C)
        patch_ptruth = torch.load(self.paths_valset[idx], weights_only=True) 
        patch = patch_ptruth.permute(2,0,1)[0:6]  # (C, H, W)
        ptruth = patch_ptruth.permute(2,0,1)[6:]  
        return patch, ptruth
    def __len__(self):
        return len(self.paths_valset)
