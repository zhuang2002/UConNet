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
import time
import cv2 as cv
from skimage.measure import compare_psnr
import scipy.io as scio

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
rain_root = ''
clean_root = ''
def test_single(anglenet_address,derainnet_adress,paramnet_adress,img_address,clean_adress):
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
    input_image = glob.glob(
        os.path.join(img_address) + '/*.png')
    psnr_add = 0
    psnr_m = 0
    input_image = sort_humanly(input_image)
    mean_a = 0
    psnr_O = 0
    ssim_m = 0
    ssim_O = 0
    mean_time = 0
    count = 0
    for num in range(len(input_image)):
        input_image_path = input_image[num]

        clean_index = input_image_path.split('\\')[-1].split('x')[0]

        input_clean_path = clean_adress + '/' + clean_index+'.png'
        input_data = cv2.imread(input_image_path, -1)
        clean_image = cv2.imread(input_clean_path, -1)
        input_y = Image.fromarray(input_data)
        input_y = transform(input_y).cuda()
        input_y = Variable(input_y)
        clean_y = Image.fromarray(clean_image)
        clean_y = transform(clean_y)
        input_y = input_y[:3, :, :]
        clean_y = clean_y[:3, :, :]
        with torch.no_grad():
            scale = 1
            row = int((input_y.shape[1] / 128))
            col = int((input_y.shape[2] / 128))
            all = row * col
            r_pad = torch.nn.ReflectionPad2d(
                padding=(-int(input_y.shape[2] % 128), 0, -int(input_y.shape[1] % 128), 0))
            temp = r_pad(input_y.unsqueeze(0))
            patch = torch.reshape(
                torch.reshape(temp.permute(0, 2, 1, 3), (row, 128, 3, temp.shape[3])).permute(0, 3, 1, 2),
                (all, 128, 128, 3)).permute(0, 3, 2, 1)
            p = 31
            if p > all:
                p = all
            index_list = random.sample(range(all), p)
            input_patch = patch[index_list,:,:,:]
            param_y = torch.mean(input_patch, dim=1)
            t0 = time.time()
            out_angle = AngleNet(input_patch)
            angle_predicted = (out_angle * 120 - 60) / 180 * 3.1415926535897
            angle_predicted = torch.reshape(angle_predicted, (angle_predicted.shape[0], 1))

            out1_temp, out2_temp, out3_temp = DerainNet1(param_y.unsqueeze(1).detach(), None, angle_predicted.detach(),
                                                         None, None, None, False)
            input_feature = torch.cat((out1_temp, out2_temp, out3_temp), dim=1).cuda()
            a = paramNet(input_feature.detach()).permute(1, 0)
            a = torch.mean(a, dim=1).unsqueeze(1)

            mean_a+=a/len(input_image)
            out_angle = out_angle[:]
            out_angle, aa = out_angle.sort(0)
            angle = (out_angle[int(p / 2)] * 120 - 60) / 180 * 3.1415926535897
            angle = angle.squeeze(2)

            a = a.expand(3,3)
            angle = angle.expand(3,1)
            derain_out = DerainNet(input_y.unsqueeze(0).permute(1,0,2,3), a, angle, False).permute(1,0,2,3)
            t1 = time.time()
            if count!=0:
                mean_time += (t1-t0)/(len(input_image))
            count = count+1
            derain_out[derain_out>input_y.unsqueeze(0)] = input_y.unsqueeze(0)[derain_out>input_y.unsqueeze(0)]

            psnr_m +=batch_PSNR(clean_y.unsqueeze(0).detach().cpu(),derain_out.detach().cpu())/len(input_image)
            psnr_O +=batch_PSNR(clean_y.unsqueeze(0).detach().cpu(),input_y.unsqueeze(0).detach().cpu())/len(input_image)
            ssim_m +=batch_SSIM(clean_y.unsqueeze(0).detach().cpu(),derain_out.detach().cpu())/len(input_image)
            ssim_O +=batch_SSIM(clean_y.unsqueeze(0).detach().cpu(),input_y.unsqueeze(0).detach().cpu())/len(input_image)
            if num%20==0:
                torch.cuda.empty_cache()
    return ssim_m, psnr_m, psnr_O,ssim_O,mean_time

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def sort_humanly(v_list):
    return sorted(v_list, key=str2int)
for i in range(1):
    ssim_m,psnr_m,psnr_o,ssim_o,mean_time = test_single(r'.\result_model\AngleNet/AngleNet.pth',
                    r'.\result_model/UConNet/UconNet_1.pth',
                               r'.\result_model\ParamNet\ParamNet_1.pth',
                    rain_root,
                    clean_root)
    # ssim_m,psnr_m,psnr_o,ssim_o,mean_time = test_single(r'.\result_model\AngleNet/AngleNet.pth',
    #                 r'.\result_model/UConNet/UconNet_2.pth',
    #                            r'.\result_model\ParamNet\ParamNet_2.pth',
    #                 rain_root,
    #                 clean_root)

    print('ssim:：',ssim_m)
    print('psnr:',psnr_m)
    print('psnr_o:', psnr_o)
    print('ssim_o:', ssim_o)
    print('mean_time:',mean_time)


