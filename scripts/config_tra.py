## author: xin luo, 
## created: 2023.4.2
## modified: 2025.12.10
## Configure parameters for the model training

import torch
import torch.nn as nn
# from dataloader.img_aug import rotate, flip, torch_noise, colorjitter, numpy2tensor
from glob import glob


#### ------------- Model -------------
model_name = 'watnet'  ### option: deeplabv3plus_mobilev2, unet, deeplabv3plus, watnet

#### ------------- Directories/files -------------
dir_scene = 'data/s2-dset/s2-scene'
dir_truth = 'data/s2-dset/s2-truth'
paths_scene = sorted(glob(dir_scene + '/*_nor.tif'))
paths_truth = [dir_truth + '/' + 
               path.split('/')[-1].replace('_6Bands', '').replace('_nor', '_Truth') \
               for path in paths_scene]
### --- train data
paths_scene_tra = paths_scene[20:]        ## scenes for training (excluding the first 20 validation scenes)
paths_truth_tra = paths_truth[20:]        ## truth for training
### --- validation data 
## validation data
dir_valset = 'data/s2-dset/valset'
### --- path to save
path_weights_save = 'model/trained/' + model_name+'_weights.pth'
path_metrics_save = 'model/trained/' + model_name + '_metrics.csv'
### --- number of bands of the image.
num_bands = 6

#### ------------- Training parameters -------------
patch_size = 256    ###  
num_thread_data_load = 40
num_epoch = 200

min_img, max_img = 0, 10000   ## used for s2 image normalization
lr = 0.0002                   ## if use lr_scheduler;
batch_size_tra = 4           ## 
batch_size_val = 4           ## 
loss_bce = nn.BCELoss()       ## selected for binary classification

# ## ------------- Data tranform/augmentation -------------
# transforms_tra = [
#         colorjitter(prob=0.25, alpha=0.05, beta=0.05),    # numpy-based, !!!beta should be small（防止过度变换） 颜色变换
#         rotate(prob=0.25),           # numpy-based 旋转
#         flip(prob=0.25),             # numpy-based 翻转
#         numpy2tensor(),              # numpy转tensor
#         torch_noise(prob=0.25, std_min=0, std_max=0.1),      # tensor-based 噪声变换
#             ]


