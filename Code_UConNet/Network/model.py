#coding=utf-8

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from lib.utils import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+2
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


class Residual_2(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual_2, self).__init__()
        #两个3*3的卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride).cuda()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1).cuda()
        #1*1的卷积保证维度一致
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride).cuda()
        else:
            self.conv3 = None
        #BN层
        self.bn1 = nn.BatchNorm2d(out_channels).cuda()
        self.bn2 = nn.BatchNorm2d(out_channels).cuda()
    def forward(self, X):
        Y = self.conv1(X)
        Y = self.bn1(Y)
        Y = torch.nn.functional.relu(Y)

        Y = self.conv2(Y)
        Y = self.bn2(Y)

        if self.conv3:
            X = self.conv3(X)
        return torch.nn.functional.relu(Y + X)


class FCNnet(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2,n_hidden_3, out_dim):
        super(FCNnet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1).cuda()
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2).cuda()
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3).cuda()
        self.layer4 = nn.Linear(n_hidden_3, out_dim).cuda()

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer3(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer4(x)
        # x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.sigmoid(x)
        return x


class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.res1 = Residual_2(1,8,True)
        self.res2 = Residual_2(8,8,True)
        self.res3 = Residual_2(8,16,True)
        self.res4 = Residual_2(16, 16, True)
    def forward(self, X):
        out = self.res1(X)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        return out

class Encode_B(nn.Module):
    def __init__(self):
        super(Encode_B, self).__init__()
        self.res1 = Residual_2(1,8,True)
        self.res2 = Residual_2(8,16,True)
        self.res3 = Residual_2(16, 16, True)
    def forward(self, X):
        out = self.res1(X)
        out = self.res2(out)
        out = self.res3(out)
        return out

class Midcode(nn.Module):
    def __init__(self):
        super(Midcode, self).__init__()
        self.res1 = Residual_2(16,32,True)
        self.res2 = Residual_2(32,32,True)
        self.res3 = Residual_2(32,32,True)
        # self.res4 = Residual_2(32, 32, True)
        self.res4 = Residual_2(32, 16, True)
    def forward(self, X):
        out = self.res1(X)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        # out = self.res5(out)
        return out

class Midcode_B(nn.Module):
    def __init__(self):
        super(Midcode_B, self).__init__()
        self.res1 = Residual_2(16,24,True)
        self.res2 = Residual_2(24,24,True)
        self.res3 = Residual_2(24,24,True)
        self.res4 = Residual_2(24,12, True)
    def forward(self, X):
        out = self.res1(X)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        return out

class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        self.res1 = Residual_2(48,64,True)
        self.res2 = Residual_2(64, 32, True)
        self.res3 = Residual_2(32,32,True)
        self.res4 = Residual_2(32, 32, True)
        self.res5 = Residual_2(32,16,True)
        self.res6 = Residual_2(16,8, True)
        self.res7 = Residual_2(8,1, True)
    def forward(self, X):
        out = self.res1(X)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        return out
class Decode_B(nn.Module):
    def __init__(self):
        super(Decode_B, self).__init__()
        self.res1 = Residual_2(52,64,True)
        self.res2 = Residual_2(64, 32, True)
        self.res3 = Residual_2(32,32,True)
        self.res4 = Residual_2(32,16,True)
        self.res5 = Residual_2(16,8, True)
        self.res6 = Residual_2(8,1, True)
    def forward(self, X):
        out = self.res1(X)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        return out
class down(nn.Module):
    def __init__(self, in_ch, out_ch,stride):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(stride),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch).cuda(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch).cuda(),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class Angle_Net(nn.Module):
    def __init__(self,in_channels):
        super(Angle_Net, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0).cuda()
        self.code = double_conv(in_channels,32).cuda()

        self.down1 = down(32,64,2).cuda()
        self.down2 = down(64,128,4).cuda()
        self.down3 = down(128,64,4).cuda()
        self.down4 = down(64,32,4).cuda()
        self.angle_layer = nn.Conv2d(32,1,kernel_size=3,padding=1).cuda()
    def forward(self,input):
        x = input
        x1 = self.conv1x1(x)    #(n,c,128,128)
        x1 = self.code(x1)      #(n,c,128,128)
        x2 = self.down1(x1)     #(n,c,64,64)
        x3 = self.down2(x2)     #(n,c,16,16)
        x4 = self.down3(x3)     #(n,c,4,4)
        x5 = self.down4(x4)     #(n,c,1,1)
        out_put = self.angle_layer(x5)
        return out_put

class derain_Net(nn.Module):
    def __init__(self):
        super(derain_Net, self).__init__()
        self.Encode = Encode()
        self.Midcode1 = Midcode()
        self.Midcode2 = Midcode()
        self.Midcode3 = Midcode()
        self.Decode = Decode()

    def forward(self,X,a,theta,istrain):

        out = self.Encode(X)
        # out1 = self.Midcode1(out)
        if istrain:
            ReflectionPad = nn.ReflectionPad2d(padding=(28, 28, 28, 28))
            out_pad = ReflectionPad(out)
            mat = get_rotate_matrix(-theta)
            affine_out = F.grid_sample(out_pad, F.affine_grid(mat, out_pad.size()))
            out1_temp = self.Midcode1(affine_out)
            out2_temp = self.Midcode2(affine_out)
            out3_temp = self.Midcode3(affine_out)
            mat_T = get_rotate_matrix(theta)
            out1 = F.grid_sample(out1_temp, F.affine_grid(mat_T, out1_temp.size()))[:, :, 28:-28, 28:-28]
            out2 = F.grid_sample(out2_temp, F.affine_grid(mat_T, out2_temp.size()))[:, :, 28:-28, 28:-28]
            out3 = F.grid_sample(out3_temp, F.affine_grid(mat_T, out3_temp.size()))[:, :, 28:-28, 28:-28]
            a1 = a[0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a2 = a[1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a3 = a[2, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            out_all = torch.cat((a1 * out1, a2 * out2, a3 * out3), dim=1).cuda()
            out_final = self.Decode(out_all)
            return X-out_final
        else:
            if 1:
                alpha = theta.squeeze(1)[0]
                h = out.shape[2]
                w = out.shape[3]
                nh = torch.ceil(w*torch.abs(torch.sin(alpha))+h*torch.cos(alpha)).unsqueeze(0)
                nw = torch.ceil(w*torch.cos(alpha)+h*torch.abs(torch.sin(alpha))).unsqueeze(0)
                hw_all = torch.cat((nh,nw),dim=0)
                max_hw = torch.max(hw_all,dim=0)[0]
                pad_h = torch.ceil((max_hw-h)/2)
                pad_w = torch.ceil((max_hw-w)/2)
                ReflectionPad_fist = nn.ReflectionPad2d(padding=(int(pad_w), int(pad_w), int(pad_h), int(pad_h)))
                out_pad = ReflectionPad_fist(out)
                mat = get_rotate_matrix(-theta)
                affine_out = F.grid_sample(out_pad, F.affine_grid(mat, out_pad.size()))
                over_h = torch.floor((max_hw-nh)/2)
                over_w = torch.floor((max_hw-nw)/2)
                RReflectionPad_next = nn.ReflectionPad2d(padding=(-int(over_w), -int(over_w), -int(over_h), -int(over_h)))
                input_affine = RReflectionPad_next(affine_out)
                out1_temp1 = self.Midcode1(input_affine)
                out2_temp1 = self.Midcode2(input_affine)
                out3_temp1 = self.Midcode3(input_affine)
                ReflectionPad_next = nn.ReflectionPad2d(padding=(int(over_w), int(over_w), int(over_h), int(over_h)))
                out1_pad = ReflectionPad_next(out1_temp1)
                out2_pad = ReflectionPad_next(out2_temp1)
                out3_pad = ReflectionPad_next(out3_temp1)
                mat_T = get_rotate_matrix(theta)

                out1_temp2 = F.grid_sample(out1_pad, F.affine_grid(mat_T, out1_pad.size()))
                out2_temp2 = F.grid_sample(out2_pad, F.affine_grid(mat_T, out2_pad.size()))
                out3_temp2 = F.grid_sample(out3_pad, F.affine_grid(mat_T, out3_pad.size()))
                RReflectionPad_fist = nn.ReflectionPad2d(padding=(-int(pad_w), -int(pad_w), -int(pad_h), -int(pad_h)))
                out1 = RReflectionPad_fist(out1_temp2)
                out2 = RReflectionPad_fist(out2_temp2)
                out3 = RReflectionPad_fist(out3_temp2)
                a1 = a[0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                a2 = a[1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                a3 = a[2, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                out_all = torch.cat((a1 * out1, a2 * out2, a3 * out3), dim=1).cuda()
                out_final = self.Decode(out_all)
                return (X-out_final)
                # return out3[:,1:2,:,:]
            else:
                print("Can't batch")

class derain_Net1(nn.Module):
    def __init__(self):
        super(derain_Net1, self).__init__()
        self.Encode = Encode()
        self.Midcode1 = Midcode()
        self.Midcode2 = Midcode()
        self.Midcode3 = Midcode()
        self.Decode = Decode()
    def forward(self,X,a, theta, out1_temp,out2_temp,out3_temp,isdecode):
        if not isdecode:
            out = self.Encode(X)
            ReflectionPad = nn.ReflectionPad2d(padding=(28, 28, 28, 28))
            out_pad = ReflectionPad(out)
            # X_pad = ReflectionPad(X)
            mat = get_rotate_matrix(-theta)
            affine_out = F.grid_sample(out_pad, F.affine_grid(mat, out_pad.size()))
            # affine_X = F.grid_sample(X_pad, F.affine_grid(mat, X_pad.size()))
            out1_temp1 = self.Midcode1(affine_out)
            out2_temp1= self.Midcode2(affine_out)
            out3_temp1 = self.Midcode3(affine_out)
            return out1_temp1, out2_temp1, out3_temp1
        else:
            mat_T = get_rotate_matrix(theta)
            out1 = F.grid_sample(out1_temp, F.affine_grid(mat_T, out1_temp.size()))[:, :, 28:-28, 28:-28]
            out2 = F.grid_sample(out2_temp, F.affine_grid(mat_T, out2_temp.size()))[:, :, 28:-28, 28:-28]
            out3 = F.grid_sample(out3_temp, F.affine_grid(mat_T, out3_temp.size()))[:, :, 28:-28, 28:-28]
            a1 = a[0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a2 = a[1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a3 = a[2, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            out_all = torch.cat((a1 * out1, a2 * out2, a3 * out3), dim=1).cuda()
            out_final = self.Decode(out_all)
            return X-out_final




class feature_Net(nn.Module):
    def __init__(self,derain_net):
        super(feature_Net, self).__init__()
        self.Encode = derain_net.Encode.clone()
        self.Midcode1 = derain_net.Midcode1.clone()
        self.Midcode2 = derain_net.Midcode2.clone()
        self.Midcode3 = derain_net.Midcode3.clone()

    def forward(self,X,theta):
        out = self.Encode(X)
        ReflectionPad = nn.ReflectionPad2d(padding=(28, 28, 28, 28))
        out_pad = ReflectionPad(out)
        mat = get_rotate_matrix(-theta)
        affine_out = F.grid_sample(out_pad, F.affine_grid(mat, out_pad.size()))
        out1_temp = self.Midcode1(affine_out)
        out2_temp = self.Midcode2(affine_out)
        out3_temp = self.Midcode3(affine_out)
        return out1_temp,out2_temp,out3_temp

    def forward(self, out1_temp,out2_temp,out3_temp, a, theta):
        mat_T = get_rotate_matrix(theta)
        out1 = F.grid_sample(out1_temp, F.affine_grid(mat_T, out1_temp.size()))[:, :, 28:-28, 28:-28]
        out2 = F.grid_sample(out2_temp, F.affine_grid(mat_T, out2_temp.size()))[:, :, 28:-28, 28:-28]
        out3 = F.grid_sample(out3_temp, F.affine_grid(mat_T, out3_temp.size()))[:, :, 28:-28, 28:-28]
        a1 = a[0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        a2 = a[1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        a3 = a[2, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        out_all = torch.cat((a1 * out1, a2 * out2, a3 * out3), dim=1).cuda()
        out_final = self.Decode(out_all)
        return out_final

class decode_Net(nn.Module):
    def __init__(self,derain_net):
        super(derain_Net, self).__init__()
        self.Decode = derain_net.Decode.clone()

    def forward(self, out1_temp,out2_temp,out3_temp, a, theta):
        mat_T = get_rotate_matrix(theta)
        out1 = F.grid_sample(out1_temp, F.affine_grid(mat_T, out1_temp.size()))[:, :, 28:-28, 28:-28]
        out2 = F.grid_sample(out2_temp, F.affine_grid(mat_T, out2_temp.size()))[:, :, 28:-28, 28:-28]
        out3 = F.grid_sample(out3_temp, F.affine_grid(mat_T, out3_temp.size()))[:, :, 28:-28, 28:-28]
        a1 = a[0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        a2 = a[1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        a3 = a[2, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        out_all = torch.cat((a1 * out1, a2 * out2, a3 * out3), dim=1).cuda()
        out_final = self.Decode(out_all)
        return out_final
class AdjParam_Neto(nn.Module):
    def __init__(self):
        super(AdjParam_Neto, self).__init__()
        self.conv1x1 = nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0).cuda()
        self.res1 = Residual_2(64, 32, True)
        self.res2 = Residual_2(32, 32, True)
        # self.res3 = Residual_2(32, 32, True)#add1
        # self.res4 = Residual_2(32, 32, True)
        self.down1 = down(32, 32, 2).cuda()#64
        self.down2 = down(32, 32, 2).cuda()#32
        self.down3 = down(32, 32, 2).cuda()#16
        self.down4 = down(32, 32, 2).cuda()  # 8
        self.down5 = down(32, 32, 2).cuda()  # 4
        self.down6 = down(32, 32, 2).cuda()  # 2
        self.down7 = down(32, 32, 2).cuda()  # 1
        self.param_layer = nn.Conv2d(32, 3, kernel_size=1, padding=0).cuda()

    def forward(self, X):
        # X = X[:,:,47:-47,47:-47]
        out = self.conv1x1(X)
        out = self.res1(out)
        out = self.res2(out)
        # out = self.res3(out)
        # out = self.res4(out)
        # print(out.shape)
        out = self.down1(out)
        # print(out.shape)
        out = self.down2(out)
        # print(out.shape)
        out = self.down3(out)
        # print(out.shape)
        out = self.down4(out)
        # out = self.res5(out)

        out = self.down5(out)
        # out = self.res6(out)

        out = self.down6(out)
        # out = self.res7(out)

        out = self.down7(out)
        # print(out.shape)
        out = self.param_layer(out)
        # print(out.shape)
        out = torch.reshape(out,(out.shape[0],3))
        # out_mean = torch.sum(out,dim=1).unsqueeze(1)
        out = F.softmax(out,dim=1)

        return out
class AdjParam_Net(nn.Module):
    def __init__(self):
        super(AdjParam_Net, self).__init__()
        self.conv1x1 = nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0).cuda()
        self.down1 = down(64, 32, 2).cuda()#64
        self.down2 = down(32, 32, 2).cuda()#32
        self.down3 = down(32, 32, 2).cuda()#16
        self.down4 = down(32, 32, 2).cuda()  # 8
        self.down5 = down(32, 32, 2).cuda()  # 4
        self.down6 = down(32, 32, 2).cuda()  # 2
        self.down7 = down(32, 32, 2).cuda()  # 1
        self.param_layer = nn.Conv2d(32, 3, kernel_size=1, padding=0).cuda()

    def forward(self, X):
        out = self.conv1x1(X)
        out = self.down1(out)
        # print(out.shape)
        out = self.down2(out)
        # print(out.shape)
        out = self.down3(out)
        # print(out.shape)
        out = self.down4(out)
        # out = self.res5(out)

        out = self.down5(out)
        # out = self.res6(out)

        out = self.down6(out)
        # out = self.res7(out)

        out = self.down7(out)
        # print(out.shape)
        out = self.param_layer(out)
        # print(out.shape)
        out = torch.reshape(out,(out.shape[0],3))
        # out_mean = torch.sum(out,dim=1).unsqueeze(1)
        out = F.softmax(out,dim=1)

        return out


class video_Net(nn.Module):
    def __init__(self):
        super(video_Net, self).__init__()
        self.Encode_A = Encode_B()
        self.Encode_B = Encode_B()
        self.Midcode1 = Midcode_B()
        self.Midcode2 = Midcode_B()
        self.Midcode3 = Midcode_B()
        self.Midcode4 = Midcode()
        self.Decode = Decode_B()

    def forward(self,X,X_B,a,theta,istrain):

        out_A = self.Encode_A(X)
        out_B = self.Encode_B(X-X_B)
        out = torch.cat((out_A,out_B),dim=1)
        # out1 = self.Midcode1(out)
        if istrain:
            ReflectionPad = nn.ReflectionPad2d(padding=(28, 28, 28, 28))
            out_pad = ReflectionPad(out)
            mat = get_rotate_matrix(-theta)
            affine_out = F.grid_sample(out_pad, F.affine_grid(mat, out_pad.size()))
            out1_temp = self.Midcode1(affine_out[:,0:16,:,:])
            out2_temp = self.Midcode2(affine_out[:,0:16,:,:])
            out3_temp = self.Midcode3(affine_out[:,0:16,:,:])
            out4_temp = self.Midcode4(affine_out[:, 16:32, :, :])
            mat_T = get_rotate_matrix(theta)
            out1 = F.grid_sample(out1_temp, F.affine_grid(mat_T, out1_temp.size()))[:, :, 28:-28, 28:-28]
            out2 = F.grid_sample(out2_temp, F.affine_grid(mat_T, out2_temp.size()))[:, :, 28:-28, 28:-28]
            out3 = F.grid_sample(out3_temp, F.affine_grid(mat_T, out3_temp.size()))[:, :, 28:-28, 28:-28]
            out4 = F.grid_sample(out4_temp, F.affine_grid(mat_T, out4_temp.size()))[:, :, 28:-28, 28:-28]
            a1 = a[0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a2 = a[1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a3 = a[2, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a4 = a[3, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            out_all = torch.cat((a1 * out1, a2 * out2, a3 * out3, a4 * out4), dim=1).cuda()
            out_final = self.Decode(out_all)
            return X-out_final
        else:
            if 1:
                alpha = theta.squeeze(1)[0]
                h = out.shape[2]
                w = out.shape[3]
                nh = torch.ceil(w*torch.abs(torch.sin(alpha))+h*torch.cos(alpha)).unsqueeze(0)
                nw = torch.ceil(w*torch.cos(alpha)+h*torch.abs(torch.sin(alpha))).unsqueeze(0)
                hw_all = torch.cat((nh,nw),dim=0)
                max_hw = torch.max(hw_all,dim=0)[0]
                pad_h = torch.ceil((max_hw-h)/2)
                pad_w = torch.ceil((max_hw-w)/2)
                ReflectionPad_fist = nn.ReflectionPad2d(padding=(int(pad_w), int(pad_w), int(pad_h), int(pad_h)))
                out_pad = ReflectionPad_fist(out)
                mat = get_rotate_matrix(-theta)
                affine_out = F.grid_sample(out_pad, F.affine_grid(mat, out_pad.size()))
                over_h = torch.floor((max_hw-nh)/2)
                over_w = torch.floor((max_hw-nw)/2)

                RReflectionPad_fist = nn.ReflectionPad2d(padding=(-int(over_w), -int(over_w), -int(over_h), -int(over_h)))
                input_affine = RReflectionPad_fist(affine_out)
                out1_temp1 = self.Midcode1(input_affine[:, 0:16, :, :])
                out2_temp1 = self.Midcode2(input_affine[:,0:16,:,:])
                out3_temp1 = self.Midcode3(input_affine[:, 0:16, :, :])
                out4_temp1 = self.Midcode4(input_affine[:, 16:32, :, :])
                ReflectionPad_next = nn.ReflectionPad2d(padding=(int(over_w), int(over_w), int(over_h), int(over_h)))
                out1_pad = ReflectionPad_next(out1_temp1)
                out2_pad = ReflectionPad_next(out2_temp1)
                out3_pad = ReflectionPad_next(out3_temp1)
                out4_pad = ReflectionPad_next(out4_temp1)
                mat_T = get_rotate_matrix(theta)

                out1_temp2 = F.grid_sample(out1_pad, F.affine_grid(mat_T, out1_pad.size()))
                out2_temp2 = F.grid_sample(out2_pad, F.affine_grid(mat_T, out2_pad.size()))
                out3_temp2 = F.grid_sample(out3_pad, F.affine_grid(mat_T, out3_pad.size()))
                out4_temp2 = F.grid_sample(out4_pad, F.affine_grid(mat_T, out4_pad.size()))
                RRReflectionPad_fist = nn.ReflectionPad2d(
                    padding=(-int(pad_w), -int(pad_w), -int(pad_h), -int(pad_h)))
                out1 = RRReflectionPad_fist(out1_temp2)
                out2 = RRReflectionPad_fist(out2_temp2)
                out3 = RRReflectionPad_fist(out3_temp2)
                out4 = RRReflectionPad_fist(out4_temp2)
                a1 = a[0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                a2 = a[1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                a3 = a[2, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                a4 = a[3, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                out_all = torch.cat((a1 * out1, a2 * out2, a3 * out3, a4 * out4), dim=1).cuda()
                out_final = self.Decode(out_all)
                return (X-out_final)
            else:
                print("Can't batch")

class VideoParam_Net(nn.Module):
    def __init__(self):
        super(VideoParam_Net, self).__init__()
        self.conv1x1 = nn.Conv2d(52, 64, kernel_size=1, stride=1, padding=0).cuda()
        self.res1 = Residual_2(64, 32, True)
        self.down1 = down(32, 32, 2).cuda()#64
        self.down2 = down(32, 32, 2).cuda()#32
        self.down3 = down(32, 32, 2).cuda()#16
        self.down4 = down(32, 32, 2).cuda()  # 8
        self.down5 = down(32, 32, 2).cuda()  # 4
        self.down6 = down(32, 32, 2).cuda()  # 2
        self.down7 = down(32, 32, 2).cuda()  # 1
        self.param_layer = nn.Conv2d(32, 4, kernel_size=1, padding=0).cuda()

    def forward(self, X):
        out = self.conv1x1(X)
        out = self.res1(out)
        out = self.down1(out)
        out = self.down2(out)
        out = self.down3(out)
        out = self.down4(out)
        out = self.down5(out)
        out = self.down6(out)
        out = self.down7(out)
        out = self.param_layer(out)
        out = torch.reshape(out,(out.shape[0],4))
        out = F.softmax(out,dim=1)

        return out


class Video_Net1(nn.Module):
    def __init__(self):
        super(Video_Net1, self).__init__()
        self.Encode_A = Encode_B()
        self.Encode_B = Encode_B()
        self.Midcode1 = Midcode_B()
        self.Midcode2 = Midcode_B()
        self.Midcode3 = Midcode_B()
        self.Midcode4 = Midcode()
        self.Decode = Decode_B()
    def forward(self,X,M_X,a, theta, out1_temp,out2_temp,out3_temp,out4_temp,isdecode):
        if not isdecode:
            out_A = self.Encode_A(X)
            out_B = self.Encode_B(X-M_X)
            out = torch.cat((out_A,out_B),dim=1)
            ReflectionPad = nn.ReflectionPad2d(padding=(28, 28, 28, 28))
            out_pad = ReflectionPad(out)
            mat = get_rotate_matrix(-theta)
            affine_out = F.grid_sample(out_pad, F.affine_grid(mat, out_pad.size()))
            out1_temp1 = self.Midcode1(affine_out[:,0:16,:,:])
            out2_temp1= self.Midcode2(affine_out[:,0:16,:,:])
            out3_temp1 = self.Midcode3(affine_out[:,0:16,:,:])
            out4_temp1 = self.Midcode4(affine_out[:,16:32,:,:])
            return out1_temp1, out2_temp1, out3_temp1, out4_temp1
        else:
            mat_T = get_rotate_matrix(theta)
            out1 = F.grid_sample(out1_temp, F.affine_grid(mat_T, out1_temp.size()))[:, :, 28:-28, 28:-28]
            out2 = F.grid_sample(out2_temp, F.affine_grid(mat_T, out2_temp.size()))[:, :, 28:-28, 28:-28]
            out3 = F.grid_sample(out3_temp, F.affine_grid(mat_T, out3_temp.size()))[:, :, 28:-28, 28:-28]
            out4 = F.grid_sample(out4_temp, F.affine_grid(mat_T, out4_temp.size()))[:, :, 28:-28, 28:-28]
            a1 = a[0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a2 = a[1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a3 = a[2, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            a4 = a[3, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)
            out_all = torch.cat((a1 * out1, a2 * out2, a3 * out3,a4*out4), dim=1).cuda()
            out_final = self.Decode(out_all)
            return X-out_final