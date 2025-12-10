## author: xin luo
## creat: 2022.4.3, modify: 2025.12.10
## des: model traing with the dset(traset or full dset)
## usage: python trainer.py 
## note: the user should set configure parameters in the scripts/config_tra.py file.

import torch
import pandas as pd
import time
from glob import glob
from scripts import config_tra
from model.seg.unet import unet
from utils.utils import read_scenes
from utils.metrics import oa_binary, miou_binary
from model.seg.deeplabv3plus import deeplabv3plus
from model.seg.surface_water.watnet import watnet
from model.seg.deeplabv3plus_mobilev2 import deeplabv3plus_mobilev2
from utils.dataloader import SceneArraySet, PatchSet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(999)   # make the training replicable

'''------train step------'''
def train_step(model, loss_fn, optimizer, x, y):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y.float())
    loss.backward()
    optimizer.step()
    ### accuracy evaluation
    miou = miou_binary(pred=pred, truth=y)
    oa = oa_binary(pred=pred, truth=y)
    return loss, miou, oa

'''------validation step------'''
def val_step(model, loss_fn, x, y):
    model.eval()    ### evaluation mode
    with torch.no_grad():
        pred = model(x.float())
        loss = loss_fn(pred, y.float())
    miou = miou_binary(pred=pred, truth=y)
    oa = oa_binary(pred=pred, truth=y)
    return loss, miou, oa

'''------ train loops ------'''
def train_loops(model, loss_fn, optimizer, tra_loader, val_loader, epoches, lr_scheduler=None):
    size_tra_loader = len(tra_loader)
    size_val_loader = len(val_loader)
    tra_loss_loops, tra_oa_loops, tra_miou_loops = [], [], []
    val_loss_loops, val_oa_loops, val_miou_loops = [], [], []

    for epoch in range(epoches):
        start = time.time()
        tra_loss, val_loss = 0, 0
        tra_miou, val_miou = 0, 0
        tra_oa, val_oa = 0, 0

        '''----- 1. train the model -----'''
        for x_batch, y_batch in tra_loader:
            if isinstance(x_batch, list):   ### multiscale input
                x_batch, y_batch = [batch.to(device) for batch in x_batch], y_batch.to(device)
            else: 
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss, miou, oa = train_step(model=model, loss_fn=loss_fn, 
                                        optimizer=optimizer, x=x_batch, y=y_batch)
            tra_loss += loss.item()
            tra_miou += miou.item()
            tra_oa += oa.item()
        if lr_scheduler:
          lr_scheduler.step(tra_loss)         # if using ReduceLROnPlateau
          # lr_scheduler.step()          # if using StepLR scheduler.

        '''----- 2. validate the model -----'''
        for x_batch, y_batch in val_loader:
            if isinstance(x_batch, list):   ## multiscale input                
                x_batch, y_batch = [batch.to(device).to(dtype=torch.float32) for batch in x_batch], y_batch.to(device)    
            else:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss, miou, oa = val_step(model=model, loss_fn=loss_fn, x=x_batch, y=y_batch)
            val_loss += loss.item()
            val_miou += miou.item()
            val_oa += oa.item()

        '''------ 3. print accuracy ------'''
        tra_loss, val_loss = tra_loss/size_tra_loader, val_loss/size_val_loader
        tra_miou, val_miou = tra_miou/size_tra_loader, val_miou/size_val_loader
        tra_oa, val_oa = tra_oa/size_tra_loader, val_oa/size_val_loader
        tra_loss_loops.append(tra_loss), tra_oa_loops.append(tra_oa), tra_miou_loops.append(tra_miou)
        val_loss_loops.append(val_loss), val_oa_loops.append(val_oa), val_miou_loops.append(val_miou)

        format = 'Ep{}: Tra-> Loss:{:.3f},Oa:{:.3f},Miou:{:.3f}, Val-> Loss:{:.3f},Oa:{:.3f},Miou:{:.3f},Time:{:.1f}s'
        print(format.format(epoch+1, tra_loss, tra_oa, tra_miou, val_loss, val_oa, val_miou, time.time()-start))

    metrics = {'tra_loss':tra_loss_loops, 'tra_oa':tra_oa_loops, 'tra_miou':tra_miou_loops, 'val_loss': val_loss_loops, 'val_oa': val_oa_loops, 'val_miou': val_miou_loops}
    return metrics


if __name__ == '__main__':
    ### 1. model instantiation
    if config_tra.model_name == 'unet':
      model = unet(num_bands=config_tra.num_bands, num_classes=2).to(device)
    elif config_tra.model_name == 'deeplabv3plus':
      model = deeplabv3plus(num_bands=config_tra.num_bands, num_classes=2).to(device)
    elif config_tra.model_name == 'deeplabv3plus_mobilev2':
      model = deeplabv3plus_mobilev2(num_bands=config_tra.num_bands, num_classes=2).to(device) 
    elif config_tra.model_name == 'watnet':
      model = watnet(num_bands=config_tra.num_bands, num_classes=2).to(device) 
    print('Model name:', config_tra.model_name)

    ## Data paths 
    ### Training part of the dataset.
    paths_scene_tra, paths_truth_tra = config_tra.paths_scene_tra, config_tra.paths_truth_tra
    ### Validation part of the dataset (patch format)
    paths_patch_valset = sorted(glob(config_tra.dir_valset+'/*'))   ## validatation patches

    '''--------- 1. Data loading --------'''
    '''----- 1.1 training data loading (from scenes path) '''
    scenes_arr, truths_arr = read_scenes(paths_scene_tra, paths_truth_tra)  ## read in memory
    tra_dset = SceneArraySet(scenes_arr=scenes_arr, 
                                truths_arr=truths_arr, 
                                path_size=(256, 256))
    print('size of training data:  ', tra_dset.__len__())
    ''' ----- 1.2. validation data loading (validation patches) ------ '''
    val_dset = PatchSet(paths_valset=paths_patch_valset)
    print('size of validation data:', val_dset.__len__())
    tra_loader = torch.utils.data.DataLoader(tra_dset, 
                                             batch_size=config_tra.batch_size_tra, 
                                             num_workers=10,
                                             shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dset, 
                                             batch_size=config_tra.batch_size_val)

    ''' -------- 2. Model loading and training strategy ------- '''
    optimizer = torch.optim.Adam(model.parameters(), lr=config_tra.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                                  mode='min', factor=0.6, patience=20)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)

    ''' -------- 3. Model training for loops ------- '''
    metrics = train_loops(model=model,  
                        loss_fn=config_tra.loss_bce, 
                        optimizer=optimizer,  
                        tra_loader=tra_loader,  
                        val_loader=val_loader,  
                        epoches=config_tra.num_epoch,  
                        lr_scheduler=lr_scheduler,
                        )

    ''' -------- 4. trained model and accuracy metric saving  ------- '''
    ## model saving

    torch.save(model.state_dict(), config_tra.path_weights_save)
    print('Model weights are saved to --> ', config_tra.path_weights_save)
    ## metrics saving
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(config_tra.path_metrics_save, index=False, sep=',')
    print('Training metrics are saved to --> ', config_tra.path_metrics_save)

