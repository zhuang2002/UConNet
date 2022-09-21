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


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def sort_humanly(v_list):
    return sorted(v_list, key=str2int)

class Angle_data(udata.Dataset):
    def __init__(self, input_root, transforms, patch_size, batch_size, repeat, channel):
        self.patch_size = patch_size
        self.channel = channel
        self.transforms = transforms
        self.batch_size = batch_size
        self.repeat = repeat
        input_image = glob.glob(os.path.join(input_root) + '/*.png')
        input_image = sort_humanly(input_image)
        self.input_image = _flatten(input_image)
        self.label = []
        for i in range(len(self.input_image)):
            label = float(self.input_image[i].split('_')[-1].split('.')[0]+'.'+self.input_image[i].split('_')[-1].split('.')[1])
            if self.label != None:
                self.label.append(label)
        if batch_size!=1:
            self.length = int(batch_size*repeat)
        else:
            self.length = len(self.input_image)


    def __getitem__(self, index):
        input_index = index % len(self.input_image)
        input_image_path = self.input_image[input_index]
        input_data = cv2.imread(input_image_path, -1)

        angle_data = np.array(self.label[input_index])
        angle_data = (angle_data.reshape([1,1,1])+60)/120

        if self.patch_size != 0:
            row = np.random.randint(input_data.shape[0] - self.patch_size)
            col = np.random.randint(input_data.shape[1] - self.patch_size)
            input_data = input_data[row:row + self.patch_size, col:col + self.patch_size, :]
        if self.channel == 1:
            im_yuv = cv2.cvtColor(input_data, cv2.COLOR_RGB2YCrCb)
            input_y = im_yuv[:, :, 0]
        else:
            input_y = input_data
            im_yuv = input_data
        input_y = Image.fromarray(input_y)
        if self.transforms:
            input_y = self.transforms(input_y)
        return {'rain_data': input_y,'angle_label':angle_data}

    def __len__(self):
        return self.length

class Real_data(udata.Dataset):
    def __init__(self, input_root, transforms, patch_size, batch_size, repeat):
        self.patch_size = patch_size
        self.transforms = transforms
        self.batch_size = batch_size
        self.repeat = repeat
        input_image = glob.glob(os.path.join(input_root) + '/*.png')
        input_image = sort_humanly(input_image)
        self.input_image = _flatten(input_image)
        self.length = repeat


    def __getitem__(self, index):
        input_index = index % len(self.input_image)
        input_image_path = self.input_image[input_index]
        input_data_o = cv2.imread(input_image_path, -1)
        input_data = None
        for i in range(self.batch_size):
            row = np.random.randint(input_data_o.shape[0] - self.patch_size)
            col = np.random.randint(input_data_o.shape[1] - self.patch_size)
            input_data_temp = input_data_o[row:row + self.patch_size, col:col + self.patch_size, :]
            if i!=0:
                input_data = np.concatenate((input_data,input_data_temp),axis=0)
            else:
                input_data = input_data_temp

        input_y = Image.fromarray(input_data)
        if self.transforms:
            input_y = self.transforms(input_y)
        return {'rain_data': input_y}

    def __len__(self):
        return self.length

class rain_data(udata.Dataset):
    def __init__(self, input_root, transforms, patch_size, batch_size, repeat, channel):
        self.patch_size = patch_size
        self.channel = channel
        self.transforms = transforms
        self.batch_size = batch_size
        self.repeat = repeat
        input_image = glob.glob(os.path.join(input_root) + '/*.png')
        input_image = sort_humanly(input_image)
        self.input_image = _flatten(input_image)

        if batch_size!=1:
            self.length = int(batch_size*repeat)
        else:
            self.length = len(self.input_image)


    def __getitem__(self, index):
        # print(len(self.input_image))
        input_index = index % len(self.input_image)
        input_image_path = self.input_image[input_index]
        input_data = cv2.imread(input_image_path, -1)


        if self.patch_size != 0:
            row = np.random.randint(input_data.shape[0] - self.patch_size)
            col = np.random.randint(input_data.shape[1] - self.patch_size)
            input_data = input_data[row:row + self.patch_size, col:col + self.patch_size, :]
        if self.channel == 1:
            im_yuv = cv2.cvtColor(input_data, cv2.COLOR_RGB2YCrCb)
            input_y = im_yuv[:, :, 0]
        else:
            input_y = input_data
            im_yuv = input_data
        input_y = Image.fromarray(input_y)
        if self.transforms:
            input_y = self.transforms(input_y)
        return {'rain_data': input_y}

    def __len__(self):
        return self.length

class param_data(udata.Dataset):
    def __init__(self, rain_root, clean_root,transforms, patch_size, batch_size, repeat, channel):
        self.patch_size = patch_size
        self.channel = channel
        self.transforms = transforms
        self.batch_size = batch_size
        self.repeat = repeat
        input_image = glob.glob(os.path.join(rain_root) + '/*.png')
        input_image = sort_humanly(input_image)
        self.input_image = _flatten(input_image)

        self.clean_root = clean_root

        if batch_size!=1:
            self.length = int(batch_size*repeat)
        else:
            self.length = len(self.input_image)


    def __getitem__(self, index):
        # print(len(self.input_image))
        input_index = index % len(self.input_image)
        input_image_path = self.input_image[input_index]
        clean_index = (input_image_path.split('\\')[-1].split('n')[-2].split('x')[0])
        # print(clean_index)
        input_data = cv2.imread(input_image_path, -1)
        # print(input_image_path)
        clean_data = cv2.imread(self.clean_root+'/norain'+clean_index+'.png')



        if self.patch_size != 0:
            row = np.random.randint(input_data.shape[0] - self.patch_size)
            col = np.random.randint(input_data.shape[1] - self.patch_size)
            input_data = input_data[row:row + self.patch_size, col:col + self.patch_size, :]
            clean_data = clean_data[row:row + self.patch_size, col:col + self.patch_size, :]
        if self.channel == 1:
            im_yuv = cv2.cvtColor(input_data, cv2.COLOR_RGB2YCrCb)
            input_y = im_yuv[:, :, 0]
            im_yuv = cv2.cvtColor(clean_data, cv2.COLOR_RGB2YCrCb)
            clean_y = im_yuv[:, :, 0]
        else:
            input_y = input_data
            im_yuv = input_data
            clean_y = clean_data
        input_y = Image.fromarray(input_y)
        clean_y = Image.fromarray(clean_y)
        if self.transforms:
            input_y = self.transforms(input_y)
            clean_y = self.transforms(clean_y)
        return {'rain_data': input_y,'clean_data':clean_y}

    def __len__(self):
        return self.length

class param_data2(udata.Dataset):
    def __init__(self, rain_root, clean_root,transforms, patch_size, batch_size, repeat, channel):
        self.patch_size = patch_size
        self.channel = channel
        self.transforms = transforms
        self.batch_size = batch_size
        self.repeat = repeat
        input_image = glob.glob(os.path.join(rain_root) + '/*.png')
        input_image = sort_humanly(input_image)
        self.input_image = _flatten(input_image)

        self.clean_root = clean_root

        if batch_size!=1:
            self.length = int(batch_size*repeat)
        else:
            self.length = len(self.input_image)


    def __getitem__(self, index):
        # print(len(self.input_image))
        input_index = index % len(self.input_image)
        input_image_path = self.input_image[input_index]
        # print(input_image_path.split("\\")[-1])
        clean_index = (input_image_path.split('\\')[-1].split('_')[0])
        # print(clean_index)
        input_data = cv2.imread(input_image_path, -1)
        # print(input_image_path)
        clean_data = cv2.imread(self.clean_root+'/'+clean_index+'.png')



        if self.patch_size != 0:
            row = np.random.randint(input_data.shape[0] - self.patch_size)
            col = np.random.randint(input_data.shape[1] - self.patch_size)
            input_data = input_data[row:row + self.patch_size, col:col + self.patch_size, :]
            clean_data = clean_data[row:row + self.patch_size, col:col + self.patch_size, :]
        if self.channel == 1:
            im_yuv = cv2.cvtColor(input_data, cv2.COLOR_RGB2YCrCb)
            input_y = im_yuv[:, :, 0]
            im_yuv = cv2.cvtColor(clean_data, cv2.COLOR_RGB2YCrCb)
            clean_y = im_yuv[:, :, 0]
        else:
            input_y = input_data
            im_yuv = input_data
            clean_y = clean_data
            # print(input_y)
        input_y = Image.fromarray(input_y)
        clean_y = Image.fromarray(clean_y)
        # PV = np.random.choice([0,1])
        # PH = np.random.choice([0,1])
        # flipH = tv.transforms.RandomHorizontalFlip(PH)
        # flipV = tv.transforms.RandomVerticalFlip(PV)
        if self.transforms:
            input_y = self.transforms(input_y)
            clean_y = self.transforms(clean_y)
        # print(input_y.shape)
        # print(clean_y.shape)
        try:
            res = input_y-clean_y
        except:
            print(input_y.shape)
            print(clean_y.shape)
            print(clean_index)
        # error = torch.rand((1,128,128))
        # rand_e = torch.rand((1))*0.02+0.98
        # error[error>rand_e] = 1
        # error[error <= rand_e] = 0
        # air_light = torch.rand((1))
        # error = error*air_light
        #input_y = (clean_y+(torch.rand((1))+0.8)*res).clamp(0,1)
        input_y =input_y.clamp(0,1)
        clean_y = clean_y.clamp(0,1)
        # print(input_y.shape)
        return {'rain_data': input_y,'clean_data':clean_y}

    def __len__(self):
        return self.length

class compare_data(udata.Dataset):
    def __init__(self, clean_root,transforms, patch_size, batch_size, repeat, channel):
        self.patch_size = patch_size
        self.channel = channel
        self.transforms = transforms
        self.batch_size = batch_size
        self.repeat = repeat
        input_image = glob.glob(os.path.join(clean_root) + '/*.jpg')
        input_image = sort_humanly(input_image)
        self.input_image = _flatten(input_image)

        if batch_size!=1:
            self.length = int(batch_size*repeat)
        else:
            self.length = len(self.input_image)


    def __getitem__(self, index):
        # print(len(self.input_image))
        input_index = index % len(self.input_image)
        input_image_path = self.input_image[input_index]
        # print(clean_index)
        input_data = cv2.imread(input_image_path, -1)



        if self.patch_size != 0:
            row = np.random.randint(input_data.shape[0] - self.patch_size)
            col = np.random.randint(input_data.shape[1] - self.patch_size)
            input_data = input_data[row:row + self.patch_size, col:col + self.patch_size, :]
        if self.channel == 1:
            im_yuv = cv2.cvtColor(input_data, cv2.COLOR_RGB2YCrCb)
            input_y = im_yuv[:, :, 0]
        else:
            input_y = input_data
            im_yuv = input_data
        input_y = Image.fromarray(input_y)
        if self.transforms:
            input_y = self.transforms(input_y)
        return {'compare_data': input_y}

    def __len__(self):
        return self.length

import scipy.io as scio
import torchvision as tv
class video_data(udata.Dataset):
    def __init__(self, root,transforms, patch_size, batch_size, repeat, channel):
        self.patch_size = patch_size
        self.channel = channel
        self.transforms = transforms
        self.batch_size = batch_size
        self.repeat = repeat
        input_image = glob.glob(os.path.join(root) + '/*.npy')
        input_image = sort_humanly(input_image)
        self.input_image = _flatten(input_image)

        if batch_size!=1:
            self.length = int(batch_size*repeat)
        else:
            self.length = len(self.input_image)


    def __getitem__(self, index):
        # print(len(self.input_image))
        input_index = index % len(self.input_image)
        input_image_path = self.input_image[input_index]
        dir_data = np.load(input_image_path,allow_pickle=True)

        B_clean = dir_data.item()['B_clean']
        Rainy = dir_data.item()['Rainy']
        Med = dir_data.item()['Med']

        if self.patch_size != 0:
            row = np.random.randint(B_clean.shape[0] - self.patch_size)
            col = np.random.randint(B_clean.shape[1] - self.patch_size)
            B_clean = B_clean[row:row + self.patch_size, col:col + self.patch_size, :]
            Rainy = Rainy[row:row + self.patch_size, col:col + self.patch_size, :]
            Med = Med[row:row + self.patch_size, col:col + self.patch_size, :]
        if self.channel == 1:
            im_yuv = cv2.cvtColor(B_clean, cv2.COLOR_RGB2YCrCb)
            B_y = im_yuv[:, :, 0]
            im_yuv = cv2.cvtColor(Rainy, cv2.COLOR_RGB2YCrCb)
            R_y = im_yuv[:, :, 0]
            im_yuv = cv2.cvtColor(Med, cv2.COLOR_RGB2YCrCb)
            M_y = im_yuv[:, :, 0]
        else:
            B_y = B_clean
            R_y = Rainy
            M_y = Med
        R_y = Image.fromarray(R_y)
        B_y = Image.fromarray(B_y)
        M_y = Image.fromarray(M_y)
        ppH = np.random.choice([0,1])
        transHflip = tv.transforms.RandomHorizontalFlip(ppH)
        ppV = np.random.choice([0, 1])
        transVflip = tv.transforms.RandomVerticalFlip(ppV)
        if self.transforms:
            M_y = transVflip(transHflip(self.transforms(M_y)))
            R_y = transVflip(transHflip(self.transforms(R_y)))
            B_y = transVflip(transHflip(self.transforms(B_y)))

        M_y = M_y.clamp(0,1)
        R_y = R_y.clamp(0,1)
        B_y = B_y.clamp(0,1)

        return {'rain_data': R_y,'clean_data':B_y,'med_data':M_y}

    def __len__(self):
        return self.length

class frame_data(udata.Dataset):
    def __init__(self, root,transforms, patch_size, batch_size, repeat, channel):
        self.patch_size = patch_size
        self.channel = channel
        self.transforms = transforms
        self.batch_size = batch_size
        self.repeat = repeat
        input_image = glob.glob(os.path.join(root) + '/*.npy')
        input_image = sort_humanly(input_image)
        self.input_image = _flatten(input_image)

        if batch_size!=1:
            self.length = int(batch_size*repeat)
        else:
            self.length = len(self.input_image)


    def __getitem__(self, index):
        # print(len(self.input_image))
        input_index = index % len(self.input_image)
        input_image_path = self.input_image[input_index]
        # print(input_image_path.split("\\")[-1])
        # print(input_image_path)
        dir_data = np.load(input_image_path,allow_pickle=True)

        B_clean = dir_data.item()['B_clean']
        Rainy = dir_data.item()['Rainy']
        # Med = dir_data.item()['Med']

        if self.patch_size != 0:
            row = np.random.randint(B_clean.shape[0] - self.patch_size)
            col = np.random.randint(B_clean.shape[1] - self.patch_size)
            B_clean = B_clean[row:row + self.patch_size, col:col + self.patch_size, :]
            Rainy = Rainy[row:row + self.patch_size, col:col + self.patch_size, :]
            # Med = Med[row:row + self.patch_size, col:col + self.patch_size, :]
        if self.channel == 1:
            im_yuv = cv2.cvtColor(B_clean, cv2.COLOR_RGB2YCrCb)
            B_y = im_yuv[:, :, 0]
            im_yuv = cv2.cvtColor(Rainy, cv2.COLOR_RGB2YCrCb)
            R_y = im_yuv[:, :, 0]
            # im_yuv = cv2.cvtColor(Med, cv2.COLOR_RGB2YCrCb)
            # M_y = im_yuv[:, :, 0]
        else:
            B_y = B_clean
            R_y = Rainy
            # M_y = Med
        R_y = Image.fromarray(R_y)
        B_y = Image.fromarray(B_y)
        # M_y = Image.fromarray(M_y)
        ppH = np.random.choice([0,1])
        transHflip = tv.transforms.RandomHorizontalFlip(ppH)
        ppV = np.random.choice([0, 1])
        transVflip = tv.transforms.RandomVerticalFlip(ppV)
        if self.transforms:
            # M_y = transVflip(transHflip(self.transforms(M_y)))
            R_y = transVflip(transHflip(self.transforms(R_y)))
            B_y = transVflip(transHflip(self.transforms(B_y)))

        # M_y = M_y.clamp(0,1)
        R_y = R_y.clamp(0,1)
        B_y = B_y.clamp(0,1)

        return {'rain_data': R_y,'clean_data':B_y}

    def __len__(self):
        return self.length