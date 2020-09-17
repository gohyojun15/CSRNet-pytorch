#%%
import argparse

import numpy as np
import time
import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

from config import Config
from model import CSRNet
from dataset import create_train_dataloader,create_test_dataloader
from utils import denormalize


parser = argparse.ArgumentParser(description="generate density map for crane")
#train datasets
parser.add_argument("--train_image_root",type=str,help="image data root")
parser.add_argument("--train_image_gt_root",type=str,help="ground truth root")
parser.add_argument("--train_image_density_root",type=str,help="density map root.")
# test datasets
parser.add_argument("--test_image_root",type=str,help="image data root")
parser.add_argument("--test_image_gt_root",type=str,help="ground truth root")
parser.add_argument("--test_image_density_root",type=str,help="density map root.")
# training arguments
parser.add_argument("--lr",type=float,default=1e-5,help="learning rate")
parser.add_argument("--epoch",type=int,default=2000,help="learning epochs")
parser.add_argument("--checkpoint_root",type=str,default="./checkpoints",help="check point root")
parser.add_argument("--batch_size",type=int, default=1,help="batch size")

"""
Argument examples

--train_image_root /home/gohyojun/바탕화면/Anthroprocene/Dataset/crane
--train_image_gt_root /home/gohyojun/바탕화면/Anthroprocene/Dataset/crane_labeled
--train_image_density_root /home/gohyojun/바탕화면/Anthroprocene/Dataset/density_map

--test_image_root /home/gohyojun/바탕화면/Anthroprocene/Dataset/crane
--test_image_gt_root /home/gohyojun/바탕화면/Anthroprocene/Dataset/crane_labeled
--test_image_density_root /home/gohyojun/바탕화면/Anthroprocene/Dataset/density_map
"""


if __name__=="__main__":
    # argument parsing.
    args = parser.parse_args()
    cfg = Config(args)                                                          # configuration
    model = CSRNet().to(cfg.device)                                         # model
    criterion = nn.MSELoss(size_average=False)                              # objective
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)              # optimizer

    train_dataloader = create_train_dataloader(cfg.train_dataset_root, use_flip=True, batch_size=cfg.batch_size)
    test_dataloader  = create_test_dataloader(cfg.test_dataset_root)             # dataloader

    min_mae = sys.maxsize
    min_mae_epoch = -1
    for epoch in range(1, cfg.epochs):                          # start training
        model.train()
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):
            image = data['image'].to(cfg.device)
            gt_densitymap = data['densitymap'].to(cfg.device) * 16# todo 1/4 rescale effect때문에
            et_densitymap = model(image)                        # forward propagation
            loss = criterion(et_densitymap,gt_densitymap)       # calculate loss
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()                                     # back propagation
            optimizer.step()                                    # update network parameters
        cfg.writer.add_scalar('Train_Loss', epoch_loss/len(train_dataloader), epoch)
        model.eval()
        with torch.no_grad():
            epoch_mae = 0.0
            for i, data in enumerate(tqdm(test_dataloader)):
                image = data['image'].to(cfg.device)
                gt_densitymap = data['densitymap'].to(cfg.device) * 16 # todo 1/4 rescale effect때문에
                et_densitymap = model(image).detach()           # forward propagation
                mae = abs(et_densitymap.data.sum()-gt_densitymap.data.sum())
                epoch_mae += mae.item()
            epoch_mae /= len(test_dataloader)
            if epoch_mae < min_mae:
                min_mae, min_mae_epoch = epoch_mae, epoch
                torch.save(model.state_dict(), os.path.join(cfg.checkpoints,str(epoch)+".pth"))     # save checkpoints
            print('Epoch ', epoch, ' MAE: ', epoch_mae, ' Min MAE: ', min_mae, ' Min Epoch: ', min_mae_epoch)   # print information
            cfg.writer.add_scalar('Val_MAE', epoch_mae, epoch)
            cfg.writer.add_image(str(epoch)+'/Image', denormalize(image[0].cpu()))

            cfg.writer.add_image(str(epoch)+'/Estimate density count:'+ str('%.2f'%(et_densitymap[0].cpu().sum())),torch.clamp(et_densitymap[0] / torch.max(et_densitymap[0]),0,1))
            cfg.writer.add_image(str(epoch)+'/Ground Truth count:'+ str('%.2f'%(gt_densitymap[0].cpu().sum())), gt_densitymap[0]/torch.max(gt_densitymap[0]))
            
# %%
