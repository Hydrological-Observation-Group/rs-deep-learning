## author: xin luo
## create: 2025.6.18
## modify: 2025.12.07
## des: accuracy metric of oa(overall accuracy), miou

import torch

def oa_binary(pred, truth, device='cpu'):
    ''' des: calculate overall accuracy (2-class classification) for each batch
        input: 
            pred: (4D tensor) 
            truth: (4D tensor)
            device: 'cpu' or 'cuda'
    '''
    pred, truth = pred.to(device), truth.to(device)
    pred_bi = torch.where(pred>0.5, 
                          torch.ones(pred.shape, device=pred.device), 
                          torch.zeros(pred.shape, device=pred.device))
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    oa = area_inter/(area_pred+0.0000001)
    oa = oa.mean()
    return oa

def miou_binary(pred, truth, device='cpu'):
    ''' des: calculate miou (2-class classification) for each batch
        input: 
            pred: (4D tensor) 
            truth: (4D tensor)
            device: 'cpu' or 'cuda'
    '''
    pred, truth = pred.to(device), truth.to(device)
    pred_bi = torch.where(pred>0.5, 
                          torch.ones(pred.shape, device=pred.device),
                          torch.zeros(pred.shape, device=pred.device))
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    area_truth = torch.histc(truth.float(), bins=2, min=0, max=1)
    area_union = area_pred + area_truth - area_inter
    iou = area_inter/(area_union+0.0000001)
    miou = iou.mean()
    return miou

