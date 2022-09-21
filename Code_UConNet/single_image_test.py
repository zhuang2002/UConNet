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
from skimage.measure import compare_psnr

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

rain_root = r'.\input\image/norain-10x2.png'
clean_root = rain_root
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

    psnr_add = 0
    for num in range(1):

        input_data = cv2.imread(img_address, -1)
        clean_image = cv2.imread(clean_adress, -1)

        input_y = Image.fromarray(input_data)
        input_y = transform(input_y)
        clean_y = Image.fromarray(clean_image)
        clean_y = transform(clean_y)
        input_y = input_y[:3,:,:]
        clean_y = clean_y[:3,:,:]
        with torch.no_grad():
            scale = 1
            row = int((input_y.shape[1]/128))
            col = int((input_y.shape[2]/128))
            all = row*col
            r_pad = torch.nn.ReflectionPad2d(
                padding=(-int(input_y.shape[2] % 128), 0, -int(input_y.shape[1] % 128), 0))
            temp = r_pad(input_y.unsqueeze(0))
            patch = torch.reshape(torch.reshape(temp.permute(0,2,1,3),(row,128,3,temp.shape[3])).permute(0,3,1,2),(all,128,128,3)).permute(0,3,2,1)
            p=31
            if p>all:
                p=all
            index_list = random.sample(range(all),p)
            for idex_num in range(p):
                if idex_num==0:
                    input_patch = patch[index_list[0]:index_list[0]+1,:,:,:]
                else:
                    input_patch = torch.cat((input_patch,patch[index_list[idex_num]:index_list[idex_num]+1,:,:,:]),dim=0)

            param_y = torch.mean(input_patch, dim=1)
            out_angle = AngleNet(input_patch)
            angle_predicted = (out_angle * 120 - 60) / 180 * 3.1415926535897
            angle_predicted = torch.reshape(angle_predicted, (angle_predicted.shape[0], 1))

            out1_temp, out2_temp, out3_temp =  DerainNet1(param_y.unsqueeze(1).detach(),None,angle_predicted.detach(),None,None,None,False)
            input_feature = torch.cat((out1_temp, out2_temp, out3_temp), dim=1).cuda()
            a = paramNet(input_feature.detach()).permute(1, 0)

            a = torch.mean(a, dim=1).unsqueeze(1)
            print(a/a[0:1,:])
            out_angle = out_angle[:]
            out_angle, aa = out_angle.sort(0)
            print(out_angle[int(p/2)] * 120 - 60)
            angle = (out_angle[int(p/2)] * 120 - 60) / 180 * 3.1415926535897
            angle = angle.squeeze(2)
            image = input_y.unsqueeze(0).permute(0, 2, 3, 1).cpu()
            image = np.array(image[0, :, :, :, ])
            cv.imshow('image', cv2.resize(image, (int((np.shape(image)[1]) / scale), int((np.shape(image)[0] / scale))),
                                          interpolation=cv2.INTER_CUBIC))
            derain_out = input_y.unsqueeze(0).clone()
            for col in range(3):
                derain_out[:, col, :, :] = DerainNet(input_y.unsqueeze(0)[:, col:col + 1, :, :], a, angle, False)

            image = derain_out.cpu().permute(0, 2, 3, 1)
            image = np.array(image[0, :, :, :])
            cv.imshow('derain', cv2.resize(image, (int((np.shape(image)[1]) / scale), int((np.shape(image)[0]) / scale)),
                                           interpolation=cv2.INTER_CUBIC))
            rain = 5 * torch.abs(clean_y.unsqueeze(0) - derain_out.cpu()).permute(0, 2, 3, 1)
            rain = np.array(rain[0, :, :, :])
            cv.imshow('Rain', cv2.resize(rain, (int((np.shape(image)[1]) / scale), int((np.shape(image)[0]) / scale)),
                                         interpolation=cv2.INTER_CUBIC))
            psnr_noisy = batch_PSNR(clean_y.unsqueeze(0).detach().cpu(), input_y.unsqueeze(0).detach().cpu())
            print(psnr_noisy)
            psnr_derain = batch_PSNR(clean_y.unsqueeze(0).detach().cpu(), derain_out.detach().cpu())
            print(psnr_derain)
            print(batch_SSIM(clean_y.unsqueeze(0).detach().cpu(),derain_out.detach().cpu()))
            psnr_add += (psnr_derain - psnr_noisy)

            cv.waitKey(0)
    return psnr_add
for i in range(1):
    print(i)
    psnr_add = test_single(r'.\result_model\AngleNet/AngleNet.pth',
                r'.\result_model/UConNet/UconNet_1.pth',
                           r'.\result_model\ParamNet\ParamNet_1.pth',
                rain_root,
                clean_root)
    psnr_add = test_single(r'.\result_model\AngleNet/AngleNet.pth',
                r'.\result_model/UConNet/UconNet_2.pth',
                           r'.\result_model\ParamNet\ParamNet_2.pth',
                rain_root,
                clean_root)
    print(psnr_add)
# print(psnr_add/9)