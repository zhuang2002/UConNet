import glob
import os
import torch
from Network.model import *
import os.path
import numpy as np
import random
# import h5py
import torch
import math
import time
import cv2, re, imageio
import glob
import torch.utils.data as udata
from PIL import Image
from _tkinter import _flatten
import torchvision as tv
# import random
import cv2 as cv
from single_img import test_single
from flow_warp import backwarp
from flow_warp import warp
from flow_warp import load_image
import argparse
import sys
sys.path.append('core')
from raft import RAFT

def psnr1(img1, img2):
   mse = np.mean((img1 - img2) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(1**2/mse)
#,'a2','a3','a4','b1','b2','b3'
for type_name in ['']:
    root_R = r'.\input\video/'
    root_C = root_R
    path_file = glob.glob(root_R+'*.jpg')
    clean_file = glob.glob(root_C+'*.jpg')
    videonet_adress = r'.\result_model\UConNet-V/UConNet-V_1.pth'
    Vparamnet_adress = r'.\result_model\ParamNet-V/ParamNet-V_1.pth'
    near_frame = 2#1 2
    VideoNet = video_Net().cuda()
    VideoNet = nn.DataParallel(VideoNet).cuda()
    VideoNet.load_state_dict(torch.load(videonet_adress))
    VideoNet.eval()

    VideoNet1 = Video_Net1().cuda()
    VideoNet1 = nn.DataParallel(VideoNet1).cuda()
    VideoNet1.load_state_dict(torch.load(videonet_adress))
    VideoNet1.eval()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r'.\models/raft-sintel.pth',
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

    anglenet_address = r'.\result_model\AngleNet/AngleNet.pth'
    derainnet_adress = r'.\result_model/UConNet/UconNet_2.pth'
    paramnet_adress = r'.\result_model\ParamNet\ParamNet_NTU.pth'
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

    VparamNet = VideoParam_Net().cuda()
    VparamNet = nn.DataParallel(VparamNet).cuda()
    VparamNet.load_state_dict(torch.load(Vparamnet_adress))
    VparamNet.eval()

    last_c = None
    last_c2 = None
    if near_frame == 1:
        len_path = len(path_file) - 2
    else:
        len_path = len(path_file) - 4
    mean_psnr = 0
    mean_psnr_y = 0
    mean_ssim = 0
    record_a = torch.zeros(4, len_path)
    with torch.no_grad():
        for idex in range(len_path):
            if near_frame == 1:
                last = load_image(path_file[idex]) / 255.0
                next = load_image(path_file[idex + 2]) / 255.0
                rainy = load_image(path_file[idex + 1]) / 255.0
                B_clean = load_image(clean_file[idex + 1]) / 255.0
            else:
                last = load_image(path_file[idex + 1]) / 255.0
                next = load_image(path_file[idex + 3]) / 255.0
                last2 = load_image(path_file[idex]) / 255.0
                next2 = load_image(path_file[idex + 4]) / 255.0
                rainy = load_image(path_file[idex + 2]) / 255.0
                B_clean = load_image(clean_file[idex + 2]) / 255.0
            B_y, Cb, Cr = yCbCr2rgb(B_clean)
            B_T = B_clean.clone()
            B_y = B_y.detach().permute(1, 2, 0).cpu().numpy()
            B_clean = B_clean.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
            t0 = time.time()
            if near_frame == 1:
                de_last, temp = test_single(last, AngleNet, paramNet, DerainNet1, DerainNet)
                de_next, temp = test_single(next, AngleNet, paramNet, DerainNet1, DerainNet)
                de_rainy, angle = test_single(rainy, AngleNet, paramNet, DerainNet1, DerainNet)
                flow_l = warp(de_rainy, de_last, model)
                if idex > 0:
                    warp_Dlast = backwarp(last_c, flow_l)
                else:
                    warp_Dlast = backwarp(de_last, flow_l)
                flow_n = warp(de_rainy, de_next, model)
                warp_next = backwarp(next, flow_n)
                cat_all = torch.cat((warp_Dlast, de_rainy, warp_next), dim=0)
            else:
                de_next = next
                de_next2 = next2
                de_rainy, angle = test_single(rainy, AngleNet, paramNet, DerainNet1, DerainNet)
                if idex == 0:
                    de_last, temp = test_single(last, AngleNet, paramNet, DerainNet1, DerainNet)
                    de_last2, temp = test_single(last2, AngleNet, paramNet, DerainNet1, DerainNet)
                    flow_l = warp(de_rainy, de_last, model)
                    flow_l2 = warp(de_rainy, de_last2, model)
                    warp_Dlast = backwarp(de_last, flow_l)
                    warp_Dlast2 = backwarp(de_last2, flow_l2)
                elif idex == 1:
                    de_last = last_c
                    de_last2, temp = test_single(last2, AngleNet, paramNet, DerainNet1, DerainNet)
                    flow_l = warp(de_rainy, de_last, model)
                    flow_l2 = warp(de_rainy, de_last2, model)
                    warp_Dlast = backwarp(last_c, flow_l)
                    warp_Dlast2 = backwarp(de_last2, flow_l2)
                else:
                    de_last = last_c
                    de_last2 = last_c2
                    flow_l = warp(de_rainy, de_last, model)
                    flow_l2 = warp(de_rainy, de_last2, model)
                    warp_Dlast = backwarp(last_c, flow_l)
                    warp_Dlast2 = backwarp(last_c2, flow_l2)
                flow_n = warp(rainy, de_next, model)
                warp_next = backwarp(next, flow_n)
                flow_n2 = warp(rainy, de_next2, model)
                warp_next2 = backwarp(next2, flow_n2)
                cat_all = torch.cat((warp_Dlast2, warp_Dlast, de_rainy, warp_next, warp_next2), dim=0)
            med = torch.median(cat_all, dim=0, keepdim=True)[0]
            input_y = rainy.squeeze(0).clone()
            input_M = med.squeeze(0).clone()

            row = int((input_y.shape[1] / 128))
            col = int((input_y.shape[2] / 128))
            all = row * col
            r_pad = torch.nn.ReflectionPad2d(
                padding=(-int(input_y.shape[2] % 128), 0, -int(input_y.shape[1] % 128), 0))
            temp = r_pad(input_y.unsqueeze(0))
            tempM = r_pad(input_M.unsqueeze(0))
            patch = torch.reshape(
                torch.reshape(temp.permute(0, 2, 1, 3), (row, 128, 3, temp.shape[3])).permute(0, 3, 1, 2),
                (all, 128, 128, 3)).permute(0, 3, 2, 1)
            patchM = torch.reshape(
                torch.reshape(tempM.permute(0, 2, 1, 3), (row, 128, 3, tempM.shape[3])).permute(0, 3, 1, 2),
                (all, 128, 128, 3)).permute(0, 3, 2, 1)

            p = 31
            if p > all:
                p = all
            index_list = random.sample(range(all), p)
            input_patch = patch[index_list, :, :, :]
            input_patchM = patchM[index_list, :, :, :]

            param_y = torch.mean(input_patch, dim=1).unsqueeze(1)
            param_M = torch.mean(input_patchM, dim=1).unsqueeze(1)

            angle_predicted = angle.expand(p, 1)
            out1_temp, out2_temp, out3_temp, out4_temp = VideoNet1(param_y.detach(), param_M.detach(), None,
                                                                   angle_predicted.detach(), None, None, None, None,
                                                                   False)
            input_feature = torch.cat((out1_temp, out2_temp, out3_temp, out4_temp), dim=1).cuda()

            a = VparamNet(input_feature.detach()).permute(1, 0)
            a = torch.mean(a, dim=1).unsqueeze(1)
            record_a[:, idex] = a.squeeze(1)
            a = a.expand(4,3)
            angle = angle.expand(3,1)
            derain_out1 = VideoNet(rainy.detach().permute(1,0,2,3),med.detach().permute(1,0,2,3), a.detach(), angle.detach(), False).permute(1,0,2,3)
            t1 = time.time()
            # print(t1-t0)
            if near_frame != 1:
                last_c2 = last_c
            last_c = derain_out1.clone()

            V = derain_out1.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
            R_np = rainy.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
            cv2.imshow('derain', V)
            cv2.imshow('O', R_np)
            cv2.imshow('Res', 5 * np.abs(R_np - V))
            cv2.waitKey(1)
            current_psnr = psnr1(V,B_clean)
            D_y, Cb, Cr = yCbCr2rgb(derain_out1)

            D_T = derain_out1.clone()
            D_y = D_y.detach().permute(1, 2, 0).cpu().numpy()
            C_psnr_y = psnr1(D_y,B_y)
            C_ssim = batch_SSIM(D_T.detach().cpu(), B_T.detach().cpu(), True)
            # cv2.imwrite(r'.\output/' +
            #             path_file[idex + 2].split('\\')[-1].split('.')[0] + '.png', (V) * 255)
            mean_psnr+=current_psnr/len_path
            mean_psnr_y +=C_psnr_y/len_path
            mean_ssim +=C_ssim/len_path

    print('type:', type_name, 'avg:', mean_psnr, ' avg_y:', mean_psnr_y, ' ssim:', mean_ssim)
