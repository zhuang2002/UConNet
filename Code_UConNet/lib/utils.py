import math,random
import torch
import torch.nn as nn
import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2
import torch.nn.functional as  F
from torch.nn.modules.module import Module
from torch.autograd import Variable


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

def transform_crop(input_tensor,angle_nor):
    angle = 90 - ((angle_nor) * 120 + 60)
    x1 = torch.clone(input_tensor)
    for i in range(input_tensor.size()[0]):
        alpha = math.radians(angle[i])
        theta = torch.tensor([
            [math.cos(alpha), math.sin(-alpha), 0],
            [math.sin(alpha), math.cos(alpha), 0]
        ], dtype=torch.float).cuda()
        img = input_tensor[i,:,:,:]
        img = img.unsqueeze(0)
        N, C, H, W = img.size()
        grid = F.affine_grid(theta.unsqueeze(0), torch.Size((N, C, W, H)))
        img = F.grid_sample(img, grid)
        x1[i,:,:,:] = img.squeeze(0)
    return x1


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def img_processing(img):
  # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray = img
  if len(gray.shape)== 3:
      print("gray error")
  # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
  x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
  y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
  absX = cv2.convertScaleAbs(x)
  return absX
# def batch_PSNR(img, imclean, data_range):
#     Img = img.data.cpu().numpy().astype(np.float32)
#     Iclean = imclean.data.cpu().numpy().astype(np.float32)
#     PSNR = 0
#     for i in range(Img.shape[0]):
#         PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
#     return (PSNR/Img.shape[0])


def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    return out

class param():
    def __init__(self, image):
        if image.shape[2] > 1:
            im_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y = im_yuv[:, :, 0]
        else:
            y = image
        y = y/255
        image = y
        self.SNR = float(np.var(image)*(0.2*np.random.random_sample(1)+0.3))
        g = np.round(0.5+np.round(8*np.random.random_sample(1)))
        if np.mod(g, 2) == 0:
            g = g+1
        g2 = np.round(0.5 + np.round(1.5 * np.random.random_sample(1)))
        if np.mod(g2, 2) == 0:
            g2 = g2 + 1
        self.g_size = (g, g)
        self.g_size2 = (g2, g2)
        self.sv = 0.1+0.3*np.random.random_sample(1)
        self.sv2 = 0.1 + 0.3 * np.random.random_sample(1)
        self.angle = 45+90*np.random.random_sample(1)
        self.length = 30+20*np.random.random_sample(1)


def get_rotate_matrix(theta):
    rotate_matrix = torch.zeros((theta.shape[0], 2, 3), device=0)
    rotate_matrix[:, 0, 0] = torch.cos(theta).squeeze(1)
    rotate_matrix[:, 0, 1] = torch.sin(-theta).squeeze(1)
    rotate_matrix[:, 1, 0] = torch.sin(theta).squeeze(1)
    rotate_matrix[:, 1, 1] = torch.cos(theta).squeeze(1)
    return rotate_matrix

def yCbCr2rgb(input_im):
    r = input_im[:, 0, :, :]
    g = input_im[:, 1, :, :]
    b = input_im[:, 2, :, :]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 0.564 * (b - Y)
    Cr = 0.713 * (r - Y)
    return Y, Cb, Cr

class Angle_loss(nn.Module):
    def __init__(self):
        super(Angle_loss,self).__init__()
    def forward(self, R, kernel,a):
        R_temp = F.pad(R, pad=(1, 1, 1, 1), mode="circular")
        # print(R_temp.shape)
        # print(kernel.shape)
        loss = F.conv2d(R_temp.permute([1,0,2,3]),kernel,padding = 0,groups=R.shape[0]).permute(1,0,2,3)
        # print(loss.shape)
        # print((a*loss).shape)
        return torch.norm(a*loss, p=1)

def L1_loss(O,B,a):
    return torch.norm(a*(O-B),p=1)

def L1_loss_add(O,B,a):
    return torch.norm(a * F.relu(B - O), p=1)

def get_kernel_conv(theta):
    temp_max = torch.max(
        torch.cat((torch.sin(theta).unsqueeze(0), torch.zeros(theta.shape, device=0).unsqueeze(0).detach()), dim=0),
        dim=0)[0]
    temp_min = -torch.min(
        torch.cat((torch.sin(theta).unsqueeze(0), torch.zeros(theta.shape, device=0).unsqueeze(0).detach()), dim=0),
        dim=0)[0]
    kernel_conv = torch.zeros((theta.shape[0], 3, 3), device=0)
    kernel_conv[:, 0, 0] = (temp_min * torch.cos(theta).clone()).squeeze(1)
    kernel_conv[:, 0, 2] = (temp_max * torch.cos(theta).clone()).squeeze(1)
    kernel_conv[:, 1, 0] = (temp_min * (
                1 - torch.cos(theta)).clone()).squeeze(1)
    kernel_conv[:, 1, 2] = (temp_max * (
                1 - torch.cos(theta)).clone()).squeeze(1)
    kernel_conv[:, 0, 1] = ((1-torch.abs(torch.sin(theta))) * torch.cos(theta).clone()).squeeze(1)
    kernel_conv[:, 1, 1] = (((1-torch.abs(torch.sin(theta)))  * (1 - torch.cos(theta)) - 1).clone()).squeeze(1)
    return kernel_conv

def get_kernel_conv_T(theta):
    temp_max = torch.max(
        torch.cat((torch.sin(theta).unsqueeze(0), torch.zeros(theta.shape, device=0).unsqueeze(0).detach()), dim=0),
        dim=0)[0]
    temp_min = -torch.min(
        torch.cat((torch.sin(theta).unsqueeze(0), torch.zeros(theta.shape, device=0).unsqueeze(0).detach()), dim=0),
        dim=0)[0]
    kernel_conv = torch.zeros((theta.shape[0], 3, 3), device=0)
    kernel_conv[:, 2, 0] = (temp_min * torch.cos(theta).clone()).squeeze(1)
    kernel_conv[:, 0, 0] = (temp_max * torch.cos(theta).clone()).squeeze(1)
    kernel_conv[:, 2, 1] = (temp_min * (
                1 - torch.cos(theta)).clone()).squeeze(1)
    kernel_conv[:, 0, 1] = (temp_max * (
                1 - torch.cos(theta)).clone()).squeeze(1)
    kernel_conv[:, 1, 0] = ((1-torch.abs(torch.sin(theta))) * torch.cos(theta).clone()).squeeze(1)
    kernel_conv[:, 1, 1] = (((1-torch.abs(torch.sin(theta)))  * (1 - torch.cos(theta)) - 1).clone()).squeeze(1)
    return kernel_conv

class AffineGridGenFunction(Module):
    def __init__(self, height, width,lr=1):
        super(AffineGridGenFunction, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        t_height = np.zeros(shape=(1,self.height),dtype=float)
        t_width = np.zeros(shape=(1,self.width),dtype=float)
        temp_height = np.expand_dims(np.arange(-1, 1, 2.0/self.height),0)
        temp_width = np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0)
        t_height[0,:] = temp_height[0,0:self.height]
        t_width[0,:] = temp_width[0,0:self.width]
        self.grid[:,:,0] = np.expand_dims(np.repeat(t_height, repeats = self.width, axis = 0).T, 0)*height/width
        self.grid[:,:,1] = np.expand_dims(np.repeat(t_width, repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        #print(self.grid)

    def forward(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            output = output.cuda()

        for i in range(input1.size(0)):
                data1 = self.batchgrid.view(-1, self.height*self.width, 3)
                data2 = torch.transpose(input1, 1, 2)
                output1 = torch.bmm(data1,data2)
                output1 = output1.view(-1, self.height, self.width, 2)
                # output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output1

    def backward(self, grad_output):

        grad_input1 = torch.zeros(self.input1.size())

        if grad_output.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            grad_input1 = grad_input1.cuda()
            #print('gradout:',grad_output.size())
        grad_input1 = torch.baddbmm(grad_input1, torch.transpose(grad_output.view(-1, self.height*self.width, 2), 1,2), self.batchgrid.view(-1, self.height*self.width, 3))

        return grad_input1

class AffineGridGen(Module):
    def __init__(self, height, width, lr = 1, aux_loss = False):
        super(AffineGridGen, self).__init__()
        self.height, self.width = height, width
        self.aux_loss = aux_loss
        self.f = AffineGridGenFunction(self.height, self.width, lr=lr)
        self.lr = lr
    def forward(self, input):
        if not self.aux_loss:
            return self.f(input)
        else:
            identity = torch.from_numpy(np.array([[1,0,0], [0,1,0]], dtype=np.float32)).cuda()
            batch_identity = torch.zeros([input.size(0), 2,3]).cuda()
            for i in range(input.size(0)):
                batch_identity[i] = identity
            batch_identity = Variable(batch_identity)
            loss = torch.mul(input - batch_identity, input - batch_identity)
            loss = torch.sum(loss,1)
            # loss = torch.sum(loss,2)

            return self.f(input), loss.view(-1,1)

def transform_no_crop(input_tensor,angle_nor):
    angle = -angle_nor
    alpha = math.radians(angle[0])
    (h, w) = input_tensor.shape[2:]
    nW = math.ceil(h * math.fabs(math.sin(alpha)) + w * math.cos(alpha))
    nH = math.ceil(h * math.cos(alpha) + w * math.fabs(math.sin(alpha)))
    x1 = torch.zeros(size=(input_tensor.shape[0],input_tensor.shape[1],nH,nW))
    for i in range(input_tensor.size()[0]):
        alpha = math.radians(angle[i])
        (h,w) = input_tensor.shape[2:]
        theta = torch.tensor([
            [math.sin(-alpha), math.cos(alpha), 0],
            [math.cos(alpha), math.sin(alpha), 0]
        ], dtype=torch.float).cuda()
        theta = theta.unsqueeze(0)
        nW = math.ceil(h * math.fabs(math.sin(alpha)) + w * math.cos(alpha))
        nH = math.ceil(h * math.cos(alpha) + w * math.fabs(math.sin(alpha)))
        img = input_tensor[i, :, :, :]
        img = img.unsqueeze(0)
        g = AffineGridGen(nH, nW, aux_loss=True)
        grid_out, aux = g(theta)
        grid_out[:, :, :, 0] = grid_out[:, :, :, 0] * nW / w
        grid_out[:, :, :, 1] = grid_out[:, :, :, 1] * nW / h
        out = F.grid_sample(img, grid_out)
        x1[i, :, :, :] = out.squeeze(0)
    return x1

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def rgb2ycbcrTorch(im, only_y=True):

    im_temp = im.permute([0,2,3,1]) * 255.0  # N x H x W x C --> N x H x W x C, [0,255]
    # convert
    if only_y:
        rlt = torch.matmul(im_temp, torch.tensor([65.481, 128.553, 24.966],
                                        device=im.device, dtype=im.dtype).view([3,1])/ 255.0) + 16.0
    else:
        rlt = torch.matmul(im_temp, torch.tensor([[65.481,  -37.797, 112.0  ],
                                                  [128.553, -74.203, -93.786],
                                                  [24.966,  112.0,   -18.214]],
                                                  device=im.device, dtype=im.dtype)/255.0) + \
                                                    torch.tensor([16, 128, 128]).view([-1, 1, 1, 3])
    rlt /= 255.0
    rlt.clamp_(0.0, 1.0)
    return rlt.permute([0, 3, 1, 2])

from skimage import img_as_ubyte
def batch_SSIM(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img_as_ubyte(img.data.numpy())
    Iclean = img_as_ubyte(imclean.data.numpy())
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += calculate_ssim(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (SSIM/Img.shape[0])
def calculate_psnr(im1, im2, border=0):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
def batch_PSNR(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img_as_ubyte(img.data.numpy())
    Iclean = img_as_ubyte(imclean.data.numpy())
    PSNR = 0
    h, w = Iclean.shape[2:]
    for i in range(Img.shape[0]):
        PSNR += calculate_psnr(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (PSNR/Img.shape[0])