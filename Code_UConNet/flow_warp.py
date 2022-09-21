import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img[[2,1,0],:,:]
    return img[None].to(DEVICE)


def viz(img, img2, flo):
    al_img = backwarp(img2, flo * 1)

    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    al_img = al_img[0].permute(1, 2, 0).cpu().numpy()
    img2 = img2[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    print(np.sum(5 * np.abs(al_img[:, :, [2, 1, 0]] / 255.0 - img[:, :, [2, 1, 0]] / 255.0)))
    print(np.sum(5 * np.abs(img[:, :, [2, 1, 0]] / 255.0 - img2[:, :, [2, 1, 0]] / 255.0)))
    cv2.imshow('nowarp', 5 * np.abs(img[:, :, [2, 1, 0]] / 255.0 - img2[:, :, [2, 1, 0]] / 255.0))
    cv2.imshow('warp', 5 * np.abs(al_img[:, :, [2, 1, 0]] / 255.0 - img[:, :, [2, 1, 0]] / 255.0))
    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([tenHorizontal, tenVertical], 1).cuda()
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2)], 1)

    return torch.nn.functional.grid_sample(input=tenInput,
                                           grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='border', align_corners=True)


# end

def warp(image1,image2,model):
    image1 = image1[:,[2,1,0],:,:]*255.0
    image2 = image2[:,[2,1,0],:,:]*255.0

    with torch.no_grad():
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

    return flow_up
