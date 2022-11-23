import scipy.io
import torch

from flow_warp import load_image
from single_img import test_single
from flow_warp import backwarp
from flow_warp import warp
import cv2
import os
from Network.model import *
from scipy.io import savemat
import numpy as np
import argparse
import sys
sys.path.append('core')
from raft import RAFT
anglenet_address = r'./result_model/AngleNet/AngleNet.pth'
derainnet_adress =  r'./result_model/UConNet/UconNet_2.pth'
paramnet_adress = r'./result_model/ParamNet/ParamNet_NTU.pth'
AngleNet = Angle_Net(in_channels=3).cuda()
AngleNet = nn.DataParallel(AngleNet).cuda()
AngleNet.load_state_dict(torch.load(anglenet_address))
AngleNet.eval()

paramNet = AdjParam_Net().cuda()
paramNet = nn.DataParallel(paramNet).cuda()
paramNet.load_state_dict(torch.load(paramnet_adress))
paramNet.eval()

DerainNet1 = derain_Net1().cuda()
DerainNet1 = nn.DataParallel(DerainNet1).cuda()
DerainNet1.load_state_dict(torch.load(derainnet_adress))
DerainNet1.eval()

DerainNet = derain_Net().cuda()
DerainNet = nn.DataParallel(DerainNet).cuda()
DerainNet.load_state_dict(torch.load(derainnet_adress))
DerainNet.eval()
parser = argparse.ArgumentParser()
parser.add_argument('--model', default=r'./models/raft-sintel.pth',
                            help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()
model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model))

model = model.module
model.to('cuda')
model.eval()
idex_start = 0
count = 2937#2144
for name1 in range(7):
    for name2 in range(4):
        file_gt = r'./input/video/GT'
        file_rain = r'./input/video/input'
        file_out = r'./input/video/npy'
        gt_path = os.listdir(file_gt)
        # print(gt_path)
        Rain_path = os.listdir(file_rain)
        # print(Rain_path)
        gt_path = sorted(gt_path)
        Rain_path = sorted(Rain_path)

        Rain_img_path = os.listdir(os.path.join(file_rain +'/'+ Rain_path[name2]))
        gt_img_path = os.listdir(os.path.join(file_gt +'/'+ gt_path[name1]))

        gt_img_path = sorted(gt_img_path)
        Rain_img_path = sorted(Rain_img_path)

        len_file = 0
        if len(gt_path)>len(Rain_path):
            len_file = len(Rain_path)
        elif len(Rain_path)>=len(gt_path):
            len_file = len(gt_path)
        # print(gt_path)
        for i in range(len_file-2):
            if Rain_path[i+2]=='Thumbs.db':
                break
            if count>-1:

                with torch.no_grad():
                    idex = i+1
                    last1 = cv2.imread(file_rain +'/'+ Rain_path[name2] +'/'+ Rain_img_path[idex - 1])/255.0
                    next1 = cv2.imread(file_rain +'/'+ Rain_path[name2] +'/'+ Rain_img_path[idex + 1])/255.0
                    # next2 = cv2.imread(file_rain +'/'+ Rain_path[idex + 2])/255.0
                    Rainy = cv2.imread(file_rain +'/'+ Rain_path[name2] +'/'+ Rain_img_path[idex])/255.0
                    B_clean = cv2.imread(file_gt +'/'+ gt_path[name2] +'/'+ gt_img_path[idex])/255.0
                    last1 = torch.Tensor(last1).permute(2,0,1).unsqueeze(0).cuda()
                    # last2 = torch.Tensor(last2).permute(2, 0, 1).unsqueeze(0).cuda()
                    next1 = torch.Tensor(next1).permute(2, 0, 1).unsqueeze(0).cuda()
                    # next2 = torch.Tensor(next2).permute(2, 0, 1).unsqueeze(0).cuda()
                    Rainy = torch.Tensor(Rainy).permute(2, 0, 1).unsqueeze(0).cuda()
                    B_clean = torch.Tensor(B_clean).permute(2, 0, 1).unsqueeze(0).cuda()
                    # print(B_clean.shape)

                    de_last1,Tmp = test_single(last1,AngleNet, paramNet, DerainNet1, DerainNet)
                    de_Rainy,Tmp = test_single(Rainy,AngleNet, paramNet, DerainNet1, DerainNet)
                    de_next1,Tmp = test_single(next1,AngleNet, paramNet, DerainNet1, DerainNet)

                    flow_upl1 = warp(de_Rainy,de_last1, model)
                    warp_Dlast1 = backwarp(de_last1,flow_upl1)

                    flow_upn1 = warp(de_Rainy, de_next1, model)
                    warp_next1 = backwarp(next1, flow_upn1)

                    warp_Dlast1 = np.array(np.expand_dims(warp_Dlast1.squeeze(0).permute(1, 2, 0).cpu().numpy()*255,axis=3)).astype(np.uint8)

                    warp_next1 = np.array(np.expand_dims(warp_next1.squeeze(0).permute(1, 2, 0).cpu().numpy()*255,axis=3)).astype(np.uint8)

                    de_Rainy = np.array(np.expand_dims(de_Rainy.squeeze(0).permute(1, 2, 0).cpu().numpy()*255,axis=3)).astype(np.uint8)
                    B_clean = np.array(B_clean.squeeze(0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                    Rainy = np.array(Rainy.squeeze(0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)

                    cat_all = np.concatenate((warp_next1,de_Rainy,warp_Dlast1),axis=3)
                    Med = np.median(cat_all, axis=3).astype(np.uint8)

                    dist = {'Rainy':Rainy,
                            'B_clean':B_clean,
                            'Med':Med}
                    np.save(file_out+'/'+str(count)+'.npy',dist)
            count += 1
        idex_start += (len(Rain_path)-4)

