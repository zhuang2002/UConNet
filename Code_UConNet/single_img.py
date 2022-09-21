import torch
from Network.model import *
import os
import os.path
import numpy as np
import random
# import h5py
import torch
import cv2, re, imageio
import glob
import torch.utils.data as udata
from PIL import Image
from _tkinter import _flatten
import torchvision as tv
# import random
import cv2 as cv
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr

transform = tv.transforms.Compose(
    [#tv.transforms.Resize(128),
        tv.transforms.ToTensor()
        # tv.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
def yCbCr2rgb(input_im):
    r = input_im[:, 0, :, :]
    g = input_im[:, 1, :, :]
    b = input_im[:, 2, :, :]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 0.564 * (b - Y)
    Cr = 0.713 * (r - Y)
    return Y, Cb, Cr

def test_single(input_y,AngleNet,paramNet,DerainNet1,DerainNet):
    input_y = input_y.squeeze(0)

    with torch.no_grad():
        scale = 1
        row = int((input_y.shape[1] / 128))
        col = int((input_y.shape[2] / 128))
        all = row * col
        r_pad = torch.nn.ReflectionPad2d(
            padding=(-int(input_y.shape[2] % 128), 0, -int(input_y.shape[1] % 128), 0))
        temp = r_pad(input_y.unsqueeze(0))
        # print(int(input_y.shape[1]%128))
        patch = torch.reshape(torch.reshape(temp.permute(0, 2, 1, 3), (row, 128, 3, temp.shape[3])).permute(0, 3, 1, 2),
                              (all, 128, 128, 3)).permute(0, 3, 2, 1)
        p = 31
        if p > all:
            p = all
        index_list = random.sample(range(all), p)
        input_patch = patch[index_list, :, :, :]

        param_y = torch.mean(input_patch, dim=1)
        out_angle = AngleNet(input_patch)
        angle_predicted = (out_angle * 120 - 60) / 180 * 3.1415926535897
        angle_predicted = torch.reshape(angle_predicted, (angle_predicted.shape[0], 1))

        out1_temp, out2_temp, out3_temp = DerainNet1(param_y.unsqueeze(1).detach(), None, angle_predicted.detach(),
                                                     None, None, None, False)
        input_feature = torch.cat((out1_temp, out2_temp, out3_temp), dim=1).cuda()
        a = paramNet(input_feature.detach()).permute(1, 0)
        a = torch.mean(a, dim=1).unsqueeze(1)

        out_angle = out_angle[:]
        out_angle, aa = out_angle.sort(0)
        angle = (out_angle[int(p / 2)] * 120 - 60) / 180 * 3.1415926535897
        angle = angle.squeeze(2)

        # derain_out = input_y.unsqueeze(0).clone()
        # print(input_y.shape)
        a = a.expand(3,3)
        angle1 = angle.expand(3,1)
        derain_out = DerainNet(input_y.unsqueeze(0).permute(1,0,2,3), a, angle1, False).permute(1,0,2,3)
        # for col in range(3):
        #     derain_out[:, col, :, :] = DerainNet(input_y.unsqueeze(0)[:, col:col + 1, :, :], a, angle, False)


    return derain_out,angle

