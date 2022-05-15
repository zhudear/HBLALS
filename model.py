from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import itertools
import numpy as np
import genotypes
from collections import OrderedDict
#from network_usrnet import *#DataNet,HyPaNet
from scipy.io import loadmat
import os
from tools import utils_image as util
from torch.optim import Adam
from losses import pytorch_ssim
from losses.vgg import VGG19_Extractor
import math
class HyPaNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus(),
                nn.AdaptiveAvgPool2d((1,1)))

    def forward(self, x):
        x = self.mlp(x)
        #print('HyPaNet1', x.shape)
        #HyPaNet1 HyPaNet1 torch.Size([1, 12, 1, 1])
        #x= x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        #print('HyPaNet',x.shape)
        #HyPaNet torch.Size([1, 12, 1, 1])
        x=x+ 1e-6
        return x
class LossFunctionl1(nn.Module):
    def __init__(self):
        super(LossFunctionl1, self).__init__()
        self.l2_loss = nn.L1Loss()#nn.MSELoss()
        # self.smooth_loss = SmoothLoss()

    def forward(self, output, target):
        #print(output.shape,target.shape) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 96, 96])
        return self.l2_loss(output, target)
class LossFunctionl2(nn.Module):
    def __init__(self):
        super(LossFunctionl2, self).__init__()
        self.l2_loss = nn.MSELoss()
        # self.smooth_loss = SmoothLoss()

    def forward(self, output, target):
        #print(output.shape,target.shape) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 96, 96])
        return self.l2_loss(output, target)
class LossFunctionlcolor(nn.Module):
    def __init__(self):
        super(LossFunctionlcolor, self).__init__()
    def forward(self, image, label):
        img1 = torch.reshape(image, [3, -1])
        img2 = torch.reshape(label, [3, -1])
        clip_value = 0.999999
        norm_vec1 = torch.nn.functional.normalize(img1, p=2, dim=0)
        norm_vec2 = torch.nn.functional.normalize(img2, p=2, dim=0)
        temp = norm_vec1 * norm_vec2
        dot = temp.sum(dim=0)
        # dot=torch.clamp(dot,min,max,out=None)
        dot = torch.clamp(dot, -clip_value, clip_value)
        angle = torch.acos(dot) * (180 / math.pi)
        # print(output.shape,target.shape) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 96, 96])
        return 0.1*torch.mean(angle)
class Lossvgg(nn.Module):
    def __init__(self):
        super(LossFunctionvgg, self).__init__()
        self.vgg = VGG19_Extractor(output_layer_list=[2, 7, 16, 25])
        for v in self.vgg.parameters():
            v.requires_grad=False
        self.l1=LossFunctionl1()
        # self.smooth_loss = SmoothLoss()

    def forward(self, output, target):
        #print(output.shape,target.shape)
        output_1_vgg_1, output_1_vgg_2, output_1_vgg_3, output_1_vgg_4 =  self.vgg(output)
        output_2_vgg_1, output_2_vgg_2, output_2_vgg_3, output_2_vgg_4 =  self.vgg(target)
        loss_c_1 = self.l1(output_1_vgg_1, output_2_vgg_1)
        loss_c_2 = self.l1(output_1_vgg_2, output_2_vgg_2)
        loss_c_3 = self.l1(output_1_vgg_3, output_2_vgg_3)
        loss_c_4 = self.l1(output_1_vgg_4, output_2_vgg_4)

        loss_vgg = loss_c_1 + loss_c_2 + loss_c_3 + loss_c_4
        return loss_vgg
class LossFunctionvgg(nn.Module):
    def __init__(self):
        super(LossFunctionvgg, self).__init__()
        '''self.vgg = VGG19_Extractor(output_layer_list=[2, 7, 16, 25])
        for v in self.vgg.parameters():
            v.requires_grad=False'''
        self.l1=LossFunctionl1()
        # self.smooth_loss = SmoothLoss()

    def forward(self, output, target,vgg_model):
        #print(output.shape,target.shape)
        output_1_vgg_1, output_1_vgg_2, output_1_vgg_3, output_1_vgg_4 =  vgg_model(output)
        output_2_vgg_1, output_2_vgg_2, output_2_vgg_3, output_2_vgg_4 =  vgg_model(target)
        loss_c_1 = self.l1(output_1_vgg_1, output_2_vgg_1)
        loss_c_2 = self.l1(output_1_vgg_2, output_2_vgg_2)
        loss_c_3 = self.l1(output_1_vgg_3, output_2_vgg_3)
        loss_c_4 = self.l1(output_1_vgg_4, output_2_vgg_4)

        loss_vgg = loss_c_1 + loss_c_2 + loss_c_3 + loss_c_4
        return loss_vgg
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class SearchBlock(nn.Module):

    def __init__(self, channel, genotype):
        super(SearchBlock, self).__init__()
        self.channel = channel

        op_names, indices = zip(*genotype.normal)
        print('op_names',op_names)

        gr1=OPS[op_names[0]](self.channel, self.channel)
        gr2 = OPS[op_names[1]](self.channel, self.channel)
        gr3 = OPS[op_names[2]](self.channel, self.channel)
        gr4 = OPS[op_names[3]](self.channel, self.channel)
        self.cr = nn.Sequential(gr1, nn.BatchNorm2d(self.channel, affine=False),
                                nn.ReLU(inplace=True),
                                gr2, nn.BatchNorm2d(self.channel, affine=False),
                                nn.ReLU(inplace=True),
                                gr3, nn.BatchNorm2d(self.channel, affine=False),
                                nn.ReLU(inplace=True),
                                gr4, nn.BatchNorm2d(self.channel, affine=False))  # self.dc)
        gi1=OPS[op_names[4]](self.channel, self.channel)
        gi2=OPS[op_names[5]](self.channel, self.channel)
        gi3=OPS[op_names[6]](self.channel, self.channel)
        gi4=OPS[op_names[7]](self.channel, self.channel)
        self.ci =nn.Sequential(gi1, nn.BatchNorm2d(self.channel, affine=False),
                               nn.ReLU(inplace=True),
                                gi2, nn.BatchNorm2d(self.channel, affine=False),
                               nn.ReLU(inplace=True),
                               gi3, nn.BatchNorm2d(self.channel, affine=False),
                               nn.ReLU(inplace=True),
                               gi4, nn.BatchNorm2d(self.channel, affine=False))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputr,inputi):
        reflectance = self.cr(inputr)
        reflectance=reflectance+inputr
        reflectance=self.relu(reflectance)

        illumination = self.ci(inputi)
        illumination=illumination+inputi
        illumination = self.relu(illumination)

        ytemp = self.sigmoid(illumination)
        zero = torch.zeros_like(ytemp)
        zero += 0.01
        ytemp = torch.max(ytemp, zero)
        ytemp = torch.div(reflectance, ytemp)
        return ytemp
class IEM(nn.Module):
    def __init__(self, channel, genetype):
        super(IEM, self).__init__()
        self.channel = channel
        self.genetype = genetype

        self.cell = SearchBlock(self.channel, self.genetype)
        self.activate = nn.Sigmoid()

    def max_operation(self, x):
        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input_y, input_u, k):
        if k == 0:
            t_hat = self.max_operation(input_y)
        else:
            t_hat = self.max_operation(input_u) - 0.5 * (input_u - input_y)
        t = t_hat
        t = self.cell(t)
        t = self.activate(t)
        t = torch.clamp(t, 0.001, 1.0)
        u = torch.clamp(input_y / t, 0.0, 1.0)

        return u, t


class EnhanceNetwork(nn.Module):
    def __init__(self, iteratioin, channel, genotype):
        super(EnhanceNetwork, self).__init__()
        self.iem_nums = iteratioin
        self.channel = channel
        self.genotype = genotype

        self.iems = nn.ModuleList()
        for i in range(self.iem_nums):
            self.iems.append(IEM(self.channel, self.genotype))

    def max_operation(self, x):
        pad = nn.ConstantPad2d(1, 0)
        x = pad(x)[:, :, 1:, 1:]
        x = torch.max(x[:, :, :-1, :], x[:, :, 1:, :])
        x = torch.max(x[:, :, :, :-1], x[:, :, :, 1:])
        return x

    def forward(self, input):
        t_list = []
        u_list = []
        u = torch.ones_like(input)
        for i in range(self.iem_nums):
            u, t = self.iems[i](input, u, i)
            u_list.append(u)
            t_list.append(t)
        return u_list, t_list

class Encodero(nn.Module):
    def __init__(self):
        super(Encodero, self).__init__()
        in_channels=3
        mid_channels=16
        out_channels=32
        self.head=nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.ce1=nn.Sequential(
            nn.Conv2d(16, mid_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, mid_channels, kernel_size=3, padding=1, bias=False)
        )
        self.ce2=nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        )
        self.down1=nn.Conv2d(mid_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.ce3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        self.ce4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.down2 = nn.Conv2d(out_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), bias=False)

        '''self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))'''

    def forward(self, input):
        h, w = input.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        input = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(input)
        #
        #DenoiseNetwork torch.Size([1, 3, 128, 128]) torch.Size([1, 1, 16384, 16384])
        input=self.head(input)
        x0 =self.ce1(input)
        x0 = input + x0
        x1 = self.ce2(x0)
        #print(input.shape, x0.shape, x1.shape)
        x1=x0+x1
        x1=self.down1(x1)

        x2 = self.ce3(x1)
        x2=x2+x1
        x3 = self.ce4(x2)
        x3=x2+x3
        x3 = self.down2(x3)
        #print('HyPaLoss', input.shape,n.shape,n)#torch.Size([1, 4, 1, 1])
        #HyPaLoss torch.Size([1, 3, 16, 16]) torch.Size([1, 3, 16, 16])
        #a list, the weight of each loss function
        return x3,x1
class Decodero(nn.Module):
    def __init__(self):
        super(Decodero, self).__init__()
        in_channels = 32
        mid_channels = 16
        out_channels = 3
        self.up1=nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.cd1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )
        self.cd2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )
        self.up2= nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.cd3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        )
        self.cd4 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        )
        self.tail=nn.Conv2d(mid_channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        '''self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))'''

    def forward(self, input,x1,x0):
        #print(input.shape,x1.shape,x0.shape)
        # print('DenoiseNetwork',input.shape,b.shape)
        # DenoiseNetwork torch.Size([1, 3, 128, 128]) torch.Size([1, 1, 16384, 16384])
        #x=torch.cat([input, x1], dim=1)
        #print(input.shape,x1.shape,x0.shape)
        #torch.Size([1, 32, 48, 48]) torch.Size([1, 32, 48, 48]) torch.Size([1, 32, 96, 96])
        io=input + x1
        io = self.up1(io)

        x = self.cd1(io)
        x=x+io
        x2 = self.cd2(x)
        x2=x2+x
        io2=x2+x0
        io2 = self.up2(io2)
        x = self.cd3(io2)
        x3=x+io2
        x = self.cd4(x3)
        x = x3 + x
        x=self.tail(x)
        return x
class SearchBlock2(nn.Module):
    def __init__(self, channel, genotype):
        super(SearchBlock2, self).__init__()
        op_names, indices = zip(*genotype.normal)
        print('SearchBlock2 op_names',op_names)
        self.head = OPS[op_names[0]](channel[0], channel[1])  #
        self.conv1=OPS[op_names[1]](channel[1], channel[1])# kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv1_2=OPS[op_names[2]](channel[1], channel[1])
        self.conv2 = OPS[op_names[3]](channel[1], channel[1])  # kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2_2 = OPS[op_names[4]](channel[1], channel[1])
        self.down1=nn.Conv2d(channel[1], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)#nn.AvgPool2d(4,2,1)
#nn.AvgPool2d(4,2,1)
        self.conv3 = OPS[op_names[5]](channel[2], channel[2])
        self.conv3_2 = OPS[op_names[6]](channel[2], channel[2])
        self.conv4 = OPS[op_names[7]](channel[2], channel[2])
        self.conv4_2 = OPS[op_names[8]](channel[2], channel[2])
        self.down2 = nn.Conv2d(channel[2], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)# nn.AvgPool2d(4, 2, 1)
#nn.AvgPool2d(4, 2, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        input = self.head(input)
        x0 = self.conv1(input)
        x0 = self.relu(x0)
        x0 = self.conv1_2(x0)
        # x0 = self.ce1(input, weights[1], weights[2])
        x0 = input + x0
        x1 = self.conv2(x0)
        x1 = self.relu(x1)
        x1 = self.conv2_2(x1)
        # x1=self.ce2(x0, weights[3], weights[4])
        x1 = x1 + x0
        x1 = self.down1(x1)

        x2 = self.conv3(x1)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)

        # x2 = self.ce3(x1, weights[5], weights[6])
        x2 = x2 + x1

        x3 = self.conv4(x2)
        x3 = self.relu(x3)
        x3 = self.conv4_2(x3)
        # x3 = self.ce4(x2, weights[7], weights[8])
        x3 = x2 + x3
        x3 = self.down2(x3)

        '''x1=self.conv1(input)
        x1=self.relu(x1)
        #x1=x1+input
        x1=self.down1(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        #x2 = x2 + x1
        x2 = self.down2(x2)'''

        return x3,x1
class SearchBlock3(nn.Module):
    def __init__(self, channel, genotype):
        super(SearchBlock3, self).__init__()
        op_names, indices = zip(*genotype.normal)
        print('SearchBlock3 op_names', op_names)
        self.up1 = nn.ConvTranspose2d(channel[0], channel[0], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv5 = OPS[op_names[0]](channel[0], channel[0])  # kernel_size=(2, 2), stride=(2, 2), bias=False)
        #self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5_2 = OPS[op_names[1]](channel[0], channel[0])
        self.conv6 = OPS[op_names[2]](channel[0], channel[0])  # kernel_size=(2, 2), stride=(2, 2), bias=False)
        # self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6_2 = OPS[op_names[3]](channel[0], channel[0])
        self.up2 = nn.ConvTranspose2d(channel[0], channel[1], kernel_size=(2, 2), stride=(2, 2),
                                      bias=False)#nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = OPS[op_names[4]](channel[1], channel[1])  # kernel_size=(2, 2), stride=(2, 2), bias=False)
        # self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7_2 = OPS[op_names[5]](channel[1], channel[1])
        self.conv8 = OPS[op_names[6]](channel[1], channel[1])  # kernel_size=(2, 2), stride=(2, 2), bias=False)
        # self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8_2 = OPS[op_names[7]](channel[1], channel[1])

        self.relu = nn.ReLU(inplace=True)
        self.tail=OPS[op_names[8]](channel[1], channel[2])
    def forward(self, input,x1,x0):
        io = input + x1
        # print('SearchBlock3', input.shape, x1.shape)
        # SearchBlock3 torch.Size([1, 64, 48, 48]) torch.Size([1, 64, 48, 48])
        io = self.up1(io)
        # print('SearchBlock3', io.shape)
        x = self.conv5(io)
        x = self.relu(x)
        x = self.conv5_2(x)
        # x = x + io
        x = x + io
        x2 = self.conv6(x)
        x2 = self.relu(x2)
        x2 = self.conv6_2(x2)
        x2 = x2 + x
        io2 = x2 + x0
        io2 = self.up2(io2)
        x = self.conv7(io2)
        x = self.relu(x)
        x = self.conv7_2(x)
        x3 = x + io2
        x = self.conv8(x3)
        x = self.relu(x)
        x = self.conv8_2(x)
        x = x + x3
        x = self.tail(x)

        '''io = input + x1
        io = self.up1(io)
        x = self.conv3(io)
        x = self.relu(x)
        #x = x + io
        io2 = x + x0
        io2 = self.up2(io2)
        x = self.conv4(io2)
        x = self.relu(x)'''
        #x = x + io2
        return x
class SearchBlock2simple(nn.Module):
    def __init__(self, channel, genotype):
        super(SearchBlock2simple, self).__init__()
        op_names, indices = zip(*genotype.normal)
        print('SearchBlock2 op_names',op_names)
        self.conv1=OPS[op_names[0]](channel[0], channel[1])# kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.down1=nn.AvgPool2d(4,2,1)
        self.conv2 = OPS[op_names[1]](channel[1], channel[2])
        self.down2 = nn.AvgPool2d(4, 2, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        x1=self.conv1(input)
        x1=self.relu(x1)
        #x1=x1+input
        x1=self.down1(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        #x2 = x2 + x1
        x2 = self.down2(x2)

        return x2,x1
class SearchBlock3simple(nn.Module):
    def __init__(self, channel, genotype):
        super(SearchBlock3simple, self).__init__()
        op_names, indices = zip(*genotype.normal)
        print('SearchBlock3 op_names', op_names)
        self.conv3 = OPS[op_names[0]](channel[0], channel[1])  # kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = OPS[op_names[1]](channel[1], channel[2])
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input,x1,x0):
        io = input + x1
        io = self.up1(io)
        x = self.conv3(io)
        x = self.relu(x)
        #x = x + io
        io2 = x + x0
        io2 = self.up2(io2)
        x = self.conv4(io2)
        x = self.relu(x)
        #x = x + io2
        return x
class Encoder(nn.Module):
    def __init__(self,genotype):
        super(Encoder, self).__init__()
        in_channels=3
        mid_channels=16
        out_channels=32
        self.encode=SearchBlock2([in_channels,mid_channels,out_channels], genotype[0])
    def forward(self, input):
        '''h, w = input.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        input = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(input)'''
        return self.encode(input)
class Decoder(nn.Module):
    def __init__(self,genotype):
        super(Decoder, self).__init__()
        in_channels = 32
        mid_channels = 16
        out_channels = 3
        self.decode=SearchBlock3([in_channels,mid_channels,out_channels],genotype[0])
    def forward(self, input,x1,x0):
        return self.decode(input,x1,x0)
class DenoiseNetwork(nn.Module):

    def __init__(self, layers, channel, genotype):
        super(DenoiseNetwork, self).__init__()

        self.nrm_nums = layers
        self.channel = channel
        self.genotype = genotype
        '''self.layer=nn.Sequential(nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                 nn.BatchNorm2d(self.channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                 nn.BatchNorm2d(self.channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.trasition=nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )'''
        #self.stem = conv_layer(4, self.channel, 3)#(3, self.channel, 3)
        self.nrms = nn.ModuleList()
        for i in range(self.nrm_nums):
            self.nrms.append(SearchBlock(self.channel, genotype[i]))
        #self.activate = nn.Sequential(conv_layer(self.channel, 3, 3))

    def forward(self, input):
        #print('Denoise',input.shape)
        #  torch.Size([1, 3, 128, 128]) torch.Size([1, 1, 16384, 16384])
        # x = torch.cat((input, b), dim=1)
        '''input=self.layer(input)
        x=self.trasition(input)
        x=self.nrms[0](input,x)'''
        x=input
        # feat = self.stem(x)
        for i in range(self.nrm_nums):
            x = self.nrms[i](x,x)
            # feat = self.nrms[i](feat, weights[0])
        # n = self.activate(feat)
        # output = input - n
        # print('DenoiseNetwork', input.shape, b.shape,output.shape)
        # DenoiseNetwork torch.Size([1, 3, 96, 96]) torch.Size([1, 1, 96, 96]) torch.Size([1, 3, 96, 96])
        # DenoiseNetwork torch.Size([1, 3, 64, 64]) torch.Size([1, 1, 64, 64]) torch.Size([1, 3, 64, 64])
        return x
def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 4

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom
def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]
#import lpips
import torch.nn as nn
#from recoLoss import RecoLoss
#import VIF_loss
#from Laplaian import LapLoss
class VIFloss(nn.Module):
    def __init__(self):
        super(VIFloss,self).__init__()
    def forward(self,ref, dist):
        #0-255
        loss=VIF_loss.vifp_mscale(ref, dist)
        return 2-loss#0=255 range max is 8
class DE3(nn.Module):
    def __init__(self):
        super(DE3,self).__init__()
    def forward(self,img):
        sum=0
        #[n,3,h,w]
        # img should be (321, 481) uint8

        #img.dtype='int'
        #img=torch.mean(img,dim=1)
        #print(img.shape,img.dtype,img)#torch.float32
        temp = img.shape[1] * img.shape[2]
        for b in range(img.shape[0]):
            tmp = torch.zeros(256)
            res = 0
            for i in range(256):
                #val = len(img[img == i])
                val=((img < (i+1)) & (img >=i)).sum()

                tmp[i] = val*1.0/temp
                #print(i,val,temp,tmp[i])
                if (tmp[i] == 0):
                    #print(i)
                    res = res
                else:
                    res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
            sum=sum+8-res
            #print(b,res)
        return sum#0=255 range max is 8
class LapLoss(nn.Module):
    def __init__(self):
        super(LapLoss, self).__init__()
        ksize = 3
        #self.l2_loss = nn.L1Loss()#nn.MSELoss()
        self.kernel=torch.tensor([[0, 1, 0],  # 这个是设置的滤波，也就是卷积核
                    [1, -4, 1],
                    [0, 1, 0]],dtype=torch.float)
        self.kernel = self.kernel.view(1, 1, ksize, ksize).repeat(3, 1, 1, 1).cuda()
        self.C=3
        # self.smooth_loss = SmoothLoss()
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x):
        batch_size = x.size()[0]
        count = self._tensor_size(x)
        result=F.conv2d(x, weight=self.kernel, bias=None,
                 stride=1, padding=0, groups=self.C)
        lap=torch.pow(result,2).sum()
        #print(output.shape,target.shape) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 96, 96])
        return lap/count/ batch_size
# pthfile = r'F:\zgj\software\pipcache\alexnet-owt-4df8aa71.pth'
# pthfile = r'F:\zgj\software\pipcache\vgg19-dcbb9e9d.pth'
class Network(nn.Module):

    def __init__(self,lw,genotype,denoise_genname):
        super(Network, self).__init__()

        #self.iem_nums = 3
        self.nrm_nums = 7
        #self.enhance_channel = 3
        self.denoise_channel = 32#6
        #self.n = 4#8
        self.l1_criterion = LossFunctionl1()
        self.l2_criterion = LossFunctionl2()
        self.color_criterion = LossFunctionlcolor()
        self.ssim_criterion = pytorch_ssim.SSIM(window_size=11)  # LossFunctionssim()
        self.vgg_criterion = LossFunctionvgg()
        #self.loss_fn_alex = lpips.LPIPS(net='alex')
        # state_dict = torch.utils.model_zoo.load_url(pthfile, model_dir=pthsavefile,
        #
        #                                             map_location=None, progress=True, check_hash=False)
        # self.loss_fn_alex.load_state_dict(state_dict)
        # for k in self.loss_fn_alex.parameters():
        #     k.requires_grad=False
        #self.recoloss = RecoLoss()
        #self.discrete_entropy = DE3()
        # self.vif=
        self.tv = TVLoss()
        self.lap = LapLoss()
        # self.loe=LoeLoss()
        # self.vif = VIFloss()

        #self._denoise_criterion = DenoiseLossFunction()

        gennamee = []
        denoise_genname0 = denoise_genname[0]  # 'genotypee' #'encoders'
        denoise_genotype0 = eval("%s.%s" % (genotype, denoise_genname0))
        gennamee.append(denoise_genotype0)

        self.e = Encoder( genotype=gennamee)

        gennamed = []
        denoise_genname0 = denoise_genname[1]  # 'genotyped'
        denoise_genotype0 = eval("%s.%s" % (genotype, denoise_genname0))
        gennamed.append(denoise_genotype0)


        self.d = Decoder(genotype=gennamed)

        self.hyper_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)

        #2021.10.05 add
        self.hyper_6 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_7 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_8 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)


        #self.hyper_6 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        # 初始化
        # [0.17, 0, 0.14, 8.37 ,1.05 ,0,0.99 LOL
        #0,0,0,5.6,  0.87,0.9,0.6,0,0.87,0
        self.hyper_1.data.fill_(lw[0])  # 0.95)#0.58
        self.hyper_2.data.fill_(lw[1])  # 0.28)#0.27
        self.hyper_3.data.fill_(lw[2])  # 0.03)#0.14
        self.hyper_4.data.fill_(lw[3])  # 9.16)#9.5
        self.hyper_5.data.fill_(lw[4])  # 0.65)#0.84

        self.hyper_6.data.fill_(lw[5])
        self.hyper_7.data.fill_(lw[6])
        self.hyper_8.data.fill_(lw[7])


        #50epoch 0.025643 0.000089 0.491792 2.987078 0.676208 0.744151 0.631382 0.000016 0.719374 0.128117  train LOL 1005 train_0921s_1005LOL
        '''self.hyper_1.data.fill_(0 )#0.95)#0.58
        self.hyper_2.data.fill_(0)#0.28)#0.27
        self.hyper_3.data.fill_(1.21)#0.03)#0.14
        self.hyper_4.data.fill_(0.59)#9.16)#9.5
        self.hyper_5.data.fill_(0)#0.65)#0.84

        self.hyper_6.data.fill_(0.63)
        self.hyper_7.data.fill_(0.87)
        self.hyper_8.data.fill_(0)
        self.hyper_9.data.fill_(0.64)
        self.hyper_10.data.fill_(0)'''
        #self.hyper_6.data.fill_(0.99)#0.65)#0.84

        genname = []
        denoise_genname0 = denoise_genname[2]  # 'genotype0'
        denoise_genotype0 = eval("%s.%s" % (genotype, denoise_genname0))
        genname.append(denoise_genotype0)
        denoise_genname1 = denoise_genname[3]  # 'genotype1'
        denoise_genotype1 = eval("%s.%s" % (genotype, denoise_genname1))
        genname.append(denoise_genotype1)
        denoise_genname2 = denoise_genname[4]  # 'genotype2'
        denoise_genotype2 = eval("%s.%s" % (genotype, denoise_genname2))
        genname.append(denoise_genotype2)
        denoise_genname3 = denoise_genname[5]  # 'genotype3'
        denoise_genotype3 = eval("%s.%s" % (genotype, denoise_genname3))
        genname.append(denoise_genotype3)
        denoise_genname4 = denoise_genname[6]  # 'genotype4'
        denoise_genotype4 = eval("%s.%s" % (genotype, denoise_genname4))
        genname.append(denoise_genotype4)
        denoise_genname5 = denoise_genname[7]  # 'genotype5'
        denoise_genotype5 = eval("%s.%s" % (genotype, denoise_genname5))
        genname.append(denoise_genotype5)
        denoise_genname6 = denoise_genname[8]  # 'genotype6'
        denoise_genotype6 = eval("%s.%s" % (genotype, denoise_genname6))
        genname.append(denoise_genotype6)
        #self.enhance_net = EnhanceNetwork(iteratioin=self.iem_nums, channel=self.enhance_channel,
                                          #genotype=enhance_genotype)
        #self.p = DenoiseNetwork(5, 32)
        self.p = DenoiseNetwork(layers=self.nrm_nums, channel=self.denoise_channel, genotype=genname)#denoise_net

        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.0005,
            momentum=0.9,
            weight_decay=3e-4)
        #self._init_weights()

    def _init_weights(self):
        model_dict = torch.load('./model/denoise.pt')
        self.denoise_net.load_state_dict(model_dict)

    def forwardorg(self, input):
        u_list, t_list = self.enhance_net(input)
        u_d, noise = self.denoise_net(u_list[-1])
        u_list.append(u_d)
        return u_list, t_list
    def forward(self, x):#
        x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x)

        x1, x0 = self.e(x)
        x2 = self.p(x1)
        x3 = self.d(x2, x1, x0)
        # print('deco', x3.shape)
        x3 = pad_tensor_back(x3, pad_left, pad_right, pad_top, pad_bottom)
        #print('forward',x.shape,x0.shape,x1.shape,x2.shape,x3.shape)
        #forward torch.Size([1, 3, 400, 600]) torch.Size([1, 32, 200, 300]) torch.Size([1, 64, 100, 150]) torch.Size([1, 64, 100, 150]) torch.Size([1, 3, 400, 600])
        #forward torch.Size([1, 3, 400, 600]) torch.Size([1, 16, 400, 600]) torch.Size([1, 32, 400, 600]) torch.Size([1, 32, 400, 600]) torch.Size([1, 3, 400, 600])
        return x3
    def _loss(self, input, target,vgg_model,type):
        #print('_loss')

        output= self(input)#,reflection,Illimination1
        '''hloss_weights = []
        for i in range(self.hloss_nums):
            hloss_weights.append(F.softmax(self.alphas_loss[i], dim=-1))
        hloss = self.hloss(input,hloss_weights)'''
        #print(input.shape, target.shape, output[-1].shape, k)
        #torch.Size([1, 3, 24, 24]) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 48, 48])
        #print(len(hloss)) #16
        finalloss=[]
        if (type==1):# True:# False:#
            #print('ssimloss')
            l1loss = self.hyper_1 * self.l1_criterion(output, target)
            l2loss = self.hyper_2 * self.l2_criterion(
                output, target)
            colorloss=self.hyper_3 *self.color_criterion(output, target)
            ssimloss =self.hyper_4 * (1 - self.ssim_criterion(output, target))
            vggloss = self.hyper_5 * self.vgg_criterion(
                output, target,vgg_model)
            lpips=20*self.loss_fn_alex(output,target)
            finalloss.append(l1loss+l2loss+colorloss+ssimloss+vggloss)
            finalloss.append(l1loss)
            finalloss.append(l2loss)
            finalloss.append(colorloss)
            finalloss.append(ssimloss)
            finalloss.append(vggloss)
            finalloss.append(lpips)
        else:
            #print('loss_fn_alex')
            loss = 0
            loss=loss+20*self.loss_fn_alex(output,target)#self.hyper_1*self.hyper_2*self.hyper_3*self.hyper_4*
            finalloss.append(loss)
        #+self._criterion(output[i*3+2], target)
        #loss = self._criterion(output[-1], target)
        '''print('############')
        for v in torch.autograd.grad(finalloss[0], self.net_parameters()):
            print(v)'''
        return finalloss
    def _loss123(self, input, target):
        output= self(input)
        #print(len(output)) 2
        loss = 0
        l1loss = 0
        l2loss = 0
        ssimloss = 0
        vggloss = 0
        finalloss=[]
        for i in range(8):
            l1loss = l1loss + self.hyper_1 * self.l1_criterion(output[i], target)
            l2loss = l2loss + self.hyper_2 * self.l2_criterion(
                output[i], target)
            ssimloss = ssimloss + self.hyper_3 * (1 - self.ssim_criterion(output[i], target))
            vggloss = vggloss + self.hyper_4 * self.vgg_criterion(
                output[i], target)
        finalloss.append(l1loss + l2loss + ssimloss + vggloss)
        finalloss.append(l1loss)
        finalloss.append(l2loss)
        finalloss.append(ssimloss)
        finalloss.append(vggloss)
        return finalloss
    def _lossorg(self, input, target):
        u_list, t_listt = self(input)
        enhance_loss = self._criterion(input, u_list, t_listt)
        denoise_loss = self._denoise_criterion(u_list[-1], u_list[-2])
        return enhance_loss + denoise_loss

    '''def _enhcence_loss(self, input, target):
        u_list, t_listt = self(input)
        enhance_loss = self._criterion(input, u_list, t_listt)
        return enhance_loss

    def _denoise_loss(self, input, target):
        u_list, t_listt = self(input)
        denoise_loss = self._denoise_criterion(u_list[-1], u_list[-2])
        return denoise_loss'''
    def optimize_parametersdehaze(self, input, target, vgg_model, step):
        # torch.autograd.set_detect_anomaly(True)
        output = self(input)
        self.optimizer.zero_grad()
        #lw = [1.31, 1.19, 0, 10.67, 2.14, 1.37, 0, 1.04, 0.89, 1.16]
        finalloss = []
        l1loss = self.hyper_1 * self.l1_criterion(output, target)
        l2loss =  self.hyper_2 * self.l2_criterion(output, target)
        colorloss =0# self.hyper_3 * self.color_criterion(output, target)
        ssimloss = self.hyper_4 * (1 - self.ssim_criterion(output, target))
        vggloss = self.hyper_5 * self.vgg_criterion(output, target, vgg_model)
        lpipsloss =  self.hyper_6 * self.loss_fn_alex(output, target)
        tvloss = self.hyper_8 * self.tv(output)
        laploss = self.hyper_9 * self.lap(output)#, target)

        # vifloss=0
        img = torch.clamp(output, 0, 1)
        img = img * 255
        # print(img)
        # (R*299 + G*587 + B*114 + 500) / 1000
        # print(img.shape,img)#torch.Size([1, 3, 400, 600])
        img[:, 0:1, :, :] = img[:, 0:1, :, :] * 299
        img[:, 1:2, :, :] = img[:, 1:2, :, :] * 587
        img[:, 2:3, :, :] = img[:, 2:3, :, :] * 114
        img = torch.sum(img, dim=1)  # /1000
        img = img / 1000


        target = target * 255
        # print(img)
        # (R*299 + G*587 + B*114 + 500) / 1000
        # print(img.shape,img)#torch.Size([1, 3, 400, 600])
        target[:, 0:1, :, :] = target[:, 0:1, :, :] * 299
        target[:, 1:2, :, :] = target[:, 1:2, :, :] * 587
        target[:, 2:3, :, :] = target[:, 2:3, :, :] * 114
        target = torch.sum(target, dim=1)  # /1000
        target = target / 1000

        deloss =  0# self.hyper_7 * self.discrete_entropy(img)
        vifloss =   self.hyper_10 * self.vif(img, target)

        # lpipsloss = self.hyper_6 * self.loss_fn_alex(output, target)
        totalloss = l1loss + l2loss + ssimloss + colorloss + vggloss + lpipsloss + deloss + tvloss + laploss + vifloss  # +lpipsloss
        finalloss.append(totalloss)
        finalloss.append(l1loss)
        finalloss.append(l2loss)
        finalloss.append(colorloss)
        finalloss.append(ssimloss)
        finalloss.append(vggloss)

        finalloss.append(lpipsloss)
        finalloss.append(deloss)
        finalloss.append(tvloss)
        finalloss.append(laploss)
        finalloss.append(vifloss)
        totalloss.backward()  # retain_graph=True)
        nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return finalloss, output
    def optimize_parameters2(self, input, target,vgg_model,step):
        #torch.autograd.set_detect_anomaly(True)
        output = self(input)
        self.optimizer.zero_grad()
        finalloss = []
        l1loss = self.hyper_1 * self.l1_criterion(output, target)
        l2loss = self.hyper_2 * self.l2_criterion(output, target)
        colorloss = self.hyper_3 * self.color_criterion(output, target)
        ssimloss = self.hyper_4 * (1 - self.ssim_criterion(output, target))
        vggloss =  self.hyper_5 * self.vgg_criterion(output, target, vgg_model)


        #2021.10.05 add
        lpipsloss =0# self.hyper_6 * self.loss_fn_alex(output, target)
        tvloss = 0#self.hyper_8 * self.tv(output)
        laploss = 0#self.hyper_9 * self.lap(output)#, target)
        # self.smoothnl1(output, target)
        '''img = torch.clamp(output, 0, 1)
        img = img * 255
        # print(img)
        # (R*299 + G*587 + B*114 + 500) / 1000
        # print(img.shape,img)#torch.Size([1, 3, 400, 600])
        img[:, 0:1, :, :] = img[:, 0:1, :, :] * 299
        img[:, 1:2, :, :] = img[:, 1:2, :, :] * 587
        img[:, 2:3, :, :] = img[:, 2:3, :, :] * 114
        img = torch.sum(img, dim=1)  # /1000

        target = target * 255
        # print(img)
        # (R*299 + G*587 + B*114 + 500) / 1000
        # print(img.shape,img)#torch.Size([1, 3, 400, 600])
        target[:, 0:1, :, :] = target[:, 0:1, :, :] * 299
        target[:, 1:2, :, :] = target[:, 1:2, :, :] * 587
        target[:, 2:3, :, :] = target[:, 2:3, :, :] * 114
        target = torch.sum(target, dim=1)  # /1000
        # print(img.shape, target.shape)#torch.Size([1, 500, 333]) torch.Size([1, 500, 333])
        img = img / 1000
        target = target / 1000'''
        vifloss =0# self.hyper_10 * self.vif(img, target)



        deloss =0# self.hyper_7 * self.discrete_entropy(img)


        #lpipsloss = self.hyper_6 * self.loss_fn_alex(output, target)
        totalloss=l1loss  +l2loss+ ssimloss +colorloss+ vggloss+lpipsloss+deloss+tvloss+laploss+vifloss#+lpipsloss
        finalloss.append(totalloss)
        finalloss.append(l1loss)
        finalloss.append(l2loss)
        finalloss.append(colorloss)
        finalloss.append(ssimloss)
        finalloss.append(vggloss)

        finalloss.append(lpipsloss)
        finalloss.append(deloss)
        finalloss.append(tvloss)
        finalloss.append(laploss)
        finalloss.append(vifloss)
        totalloss.backward()#retain_graph=True)
        nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return finalloss,output
    def optimize_parametersMIT(self, input, target, vgg_model,loss_fn_alex, step):
        # torch.autograd.set_detect_anomaly(True)
        output = self(input)
        self.optimizer.zero_grad()
        # 0, 0, 0, 5.6, 0.87, 0.9, 0.6, 0, 0.87, 0
        finalloss = []
        #lw = [1.38, 1.28, 0.82, 1.23, 0, 0.74, 0, 0]
        l1loss = self.hyper_1 * self.l1_criterion(output, target)
        l2loss = self.hyper_2 * self.l2_criterion(output, target)
        colorloss = self.hyper_3 * self.color_criterion(output, target)
        ssimloss = self.hyper_4 *10* (1 - self.ssim_criterion(output, target))
        vggloss =  self.hyper_5 * self.vgg_criterion(output, target, vgg_model)
        batch=torch.mean(loss_fn_alex(output, target))
        lpipsloss =  self.hyper_6 * batch
        #print('lpipsloss',lpipsloss)
        tvloss = 0  # self.hyper_8 * self.tv(output)
        laploss =   self.hyper_8 * self.lap(output)#, target)
        totalloss = l1loss + l2loss + ssimloss + colorloss + vggloss + lpipsloss  + tvloss + laploss  # +lpipsloss
        finalloss.append(totalloss)
        finalloss.append(l1loss)
        finalloss.append(l2loss)
        finalloss.append(colorloss)
        finalloss.append(ssimloss)
        finalloss.append(vggloss)
        finalloss.append(lpipsloss)
        finalloss.append(tvloss)
        finalloss.append(laploss)
        totalloss.backward()  # retain_graph=True)
        nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return finalloss, output
    def optimize_parametersunderwater(self, input, target, vgg_model, loss_fn_alex,step):
        # torch.autograd.set_detect_anomaly(True)
        output = self(input)
        self.optimizer.zero_grad()
        #lw = [1.16, 1.10, 0.62, 10.51, 0.74, 0.77, 2.70, 0, 0, 0]
        finalloss = []
        l1loss = self.hyper_1 * self.l1_criterion(output, target)
        l2loss =  self.hyper_2 * self.l2_criterion(output, target)
        colorloss = self.hyper_3 * self.color_criterion(output, target)
        ssimloss = self.hyper_4 *10* (1 - self.ssim_criterion(output, target))
        vggloss = self.hyper_5 * self.vgg_criterion(output, target, vgg_model)
        lpipsloss =  self.hyper_6 *  loss_fn_alex(output, target)
        tvloss = 0  # self.hyper_8 * self.tv(output)
        laploss = 0  # self.hyper_9 * self.lap(output)#, target)

        # vifloss=0
        '''img = torch.clamp(output, 0, 1)
        img = img * 255
        # print(img)
        # (R*299 + G*587 + B*114 + 500) / 1000
        # print(img.shape,img)#torch.Size([1, 3, 400, 600])
        img[:, 0:1, :, :] = img[:, 0:1, :, :] * 299
        img[:, 1:2, :, :] = img[:, 1:2, :, :] * 587
        img[:, 2:3, :, :] = img[:, 2:3, :, :] * 114
        img = torch.sum(img, dim=1)  # /1000
        img = img / 1000'''


        '''target = target * 255
        # print(img)
        # (R*299 + G*587 + B*114 + 500) / 1000
        # print(img.shape,img)#torch.Size([1, 3, 400, 600])
        target[:, 0:1, :, :] = target[:, 0:1, :, :] * 299
        target[:, 1:2, :, :] = target[:, 1:2, :, :] * 587
        target[:, 2:3, :, :] = target[:, 2:3, :, :] * 114
        target = torch.sum(target, dim=1)  # /1000
        target = target / 1000'''




        # lpipsloss = self.hyper_6 * self.loss_fn_alex(output, target)
        totalloss = l1loss + l2loss + ssimloss + colorloss + vggloss + lpipsloss +  tvloss + laploss  # +lpipsloss
        finalloss.append(totalloss)
        finalloss.append(l1loss)
        finalloss.append(l2loss)
        finalloss.append(colorloss)
        finalloss.append(ssimloss)
        finalloss.append(vggloss)

        finalloss.append(lpipsloss)
        finalloss.append(tvloss)
        finalloss.append(laploss)

        totalloss.backward()  # retain_graph=True)
        nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return finalloss, output
    def optimize_parametersLOL(self, input, target, vgg_model,loss_fn_alex, step):
        # torch.autograd.set_detect_anomaly(True)
        output = self(input)
        self.optimizer.zero_grad()
        # 0, 0, 0, 5.6, 0.87, 0.9, 0.6, 0, 0.87, 0
        finalloss = []
        #lw = [1.38, 1.28, 0.82, 1.23, 0, 0.74, 0, 0]
        l1loss = self.hyper_1 * self.l1_criterion(output, target)
        l2loss = self.hyper_2 * self.l2_criterion(output, target)
        colorloss = self.hyper_3 * self.color_criterion(output, target)
        ssimloss = self.hyper_4 *10* (1 - self.ssim_criterion(output, target))
        vggloss = self.hyper_5 * self.vgg_criterion(output, target, vgg_model)
        batch=torch.mean(loss_fn_alex(output, target))
        lpipsloss =  self.hyper_6 * batch
        #print('lpipsloss',lpipsloss)
        tvloss = 0  # self.hyper_8 * self.tv(output)
        laploss = self.hyper_8 * self.lap(output)#, target)
        totalloss = l1loss + l2loss + ssimloss + colorloss + vggloss + lpipsloss  + tvloss + laploss  # +lpipsloss
        finalloss.append(totalloss)
        finalloss.append(l1loss)
        finalloss.append(l2loss)
        finalloss.append(colorloss)
        finalloss.append(ssimloss)
        finalloss.append(vggloss)
        finalloss.append(lpipsloss)
        finalloss.append(tvloss)
        finalloss.append(laploss)
        totalloss.backward()  # retain_graph=True)
        nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return finalloss, output
    def optimize_parameters(self, input, target,vgg_model,step):
        #torch.autograd.set_detect_anomaly(True)
        output = self(input)
        self.optimizer.zero_grad()
        #0, 0, 0, 5.6, 0.87, 0.9, 0.6, 0, 0.87, 0
        finalloss = []
        l1loss = self.hyper_1 * self.l1_criterion(output, target)
        l2loss = self.hyper_2 * self.l2_criterion(output, target)
        colorloss =self.hyper_3 * self.color_criterion(output, target)
        ssimloss = self.hyper_4 * (1 - self.ssim_criterion(output, target))
        vggloss =  self.hyper_5 * self.vgg_criterion(output, target, vgg_model)
        lpipsloss =0# self.hyper_6 * self.loss_fn_alex(output, target)
        tvloss = 0#self.hyper_8 * self.tv(output)
        laploss =0# self.hyper_9 * self.lap(output)#, target)

        #vifloss=0
        '''img = torch.clamp(output, 0, 1)
        img = img * 255
        # print(img)
        # (R*299 + G*587 + B*114 + 500) / 1000
        # print(img.shape,img)#torch.Size([1, 3, 400, 600])
        img[:, 0:1, :, :] = img[:, 0:1, :, :] * 299
        img[:, 1:2, :, :] = img[:, 1:2, :, :] * 587
        img[:, 2:3, :, :] = img[:, 2:3, :, :] * 114
        img = torch.sum(img, dim=1)  # /1000
        img = img / 1000


        target = target * 255
        # print(img)
        # (R*299 + G*587 + B*114 + 500) / 1000
        # print(img.shape,img)#torch.Size([1, 3, 400, 600])
        target[:, 0:1, :, :] = target[:, 0:1, :, :] * 299
        target[:, 1:2, :, :] = target[:, 1:2, :, :] * 587
        target[:, 2:3, :, :] = target[:, 2:3, :, :] * 114
        target = torch.sum(target, dim=1)  # /1000
        target = target / 1000'''

        deloss =0# self.hyper_7 * self.discrete_entropy(img)
        vifloss =0#self.hyper_10 * self.vif(img, target)
        '''if self.hyper_1==0:
            l1loss=0
        else:
            l1loss = self.hyper_1 * self.l1_criterion(output, target)
        if self.hyper_2 == 0:
            l2loss = 0
        else:
            l2loss = self.hyper_2 * self.l2_criterion(output, target)
        if self.hyper_3 == 0:
            colorloss = 0
        else:
            colorloss = self.hyper_3 * self.color_criterion(output, target)
        if self.hyper_4 == 0:
            ssimloss = 0
        else:
            ssimloss = self.hyper_4 * (1 - self.ssim_criterion(output, target))
        if self.hyper_5 == 0:
            vggloss = 0
        else:
            vggloss = self.hyper_5 * self.vgg_criterion(
            output, target, vgg_model)


        #2021.10.05 add
        if self.hyper_6 == 0:
            lpipsloss = 0
        else:
            lpipsloss = self.hyper_6 * self.loss_fn_alex(output, target)
        if self.hyper_8 == 0:
            tvloss = 0
        else:
            tvloss = self.hyper_8 * self.tv(output)
        if self.hyper_9 == 0:
            laploss = 0
        else:
            laploss = self.hyper_9 * self.lap(output, target)
        # self.smoothnl1(output, target)
        
        if self.hyper_10 == 0:
            vifloss = 0
        else:
            img = torch.clamp(output, 0, 1)
            img = img * 255
            # print(img)
            # (R*299 + G*587 + B*114 + 500) / 1000
            # print(img.shape,img)#torch.Size([1, 3, 400, 600])
            img[:, 0:1, :, :] = img[:, 0:1, :, :] * 299
            img[:, 1:2, :, :] = img[:, 1:2, :, :] * 587
            img[:, 2:3, :, :] = img[:, 2:3, :, :] * 114
            img = torch.sum(img, dim=1)  # /1000
    
            target = target * 255
            # print(img)
            # (R*299 + G*587 + B*114 + 500) / 1000
            # print(img.shape,img)#torch.Size([1, 3, 400, 600])
            target[:, 0:1, :, :] = target[:, 0:1, :, :] * 299
            target[:, 1:2, :, :] = target[:, 1:2, :, :] * 587
            target[:, 2:3, :, :] = target[:, 2:3, :, :] * 114
            target = torch.sum(target, dim=1)  # /1000
            # print(img.shape, target.shape)#torch.Size([1, 500, 333]) torch.Size([1, 500, 333])
            img = img / 1000
            target = target / 1000
            vifloss = self.hyper_10 * self.vif(img, target)


        if self.hyper_7 == 0:
            deloss = 0
        elif  self.hyper_10 != 0:
            deloss = self.hyper_7 * self.discrete_entropy(img)
        else:
            img = torch.clamp(output, 0, 1)
            img = img * 255
            # print(img)
            # (R*299 + G*587 + B*114 + 500) / 1000
            # print(img.shape,img)#torch.Size([1, 3, 400, 600])
            img[:, 0:1, :, :] = img[:, 0:1, :, :] * 299
            img[:, 1:2, :, :] = img[:, 1:2, :, :] * 587
            img[:, 2:3, :, :] = img[:, 2:3, :, :] * 114
            img = torch.sum(img, dim=1)  # /1000
            img = img / 1000
            deloss = self.hyper_7 * self.discrete_entropy(img)'''


        #lpipsloss = self.hyper_6 * self.loss_fn_alex(output, target)
        totalloss=l1loss  +l2loss+ ssimloss +colorloss+ vggloss+lpipsloss+deloss+tvloss+laploss+vifloss#+lpipsloss
        finalloss.append(totalloss)
        finalloss.append(l1loss)
        finalloss.append(l2loss)
        finalloss.append(colorloss)
        finalloss.append(ssimloss)
        finalloss.append(vggloss)

        finalloss.append(lpipsloss)
        finalloss.append(deloss)
        finalloss.append(tvloss)
        finalloss.append(laploss)
        finalloss.append(vifloss)
        totalloss.backward()#retain_graph=True)
        nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return finalloss,output
    def optimizerorg(self, input, target, step):
        #torch.autograd.set_detect_anomaly(True)

        u_list, t_listt = self(input)
        self.enhancement_optimizer.zero_grad()
        enhancement_loss = self._criterion(input, u_list, t_listt)
        enhancement_loss.backward(retain_graph=True)

        nn.utils.clip_grad_norm(self.enhance_net.parameters(), 5)
        self.enhancement_optimizer.step()

        denoise_loss = 0
        if step % 50 == 0:
            self.denoise_optimizer.zero_grad()

            '''a=u_list[-1]
            b=u_list[-2]
            #print(a.shape,b.shape)#torch.Size([1, 3, 24, 24]) torch.Size([1, 3, 24, 24])
            a=a.mean()
            b=b.mean()
            b.backward()
            a.backward()'''
            a=u_list[-1].clone()
            b= u_list[-2].clone()
            denoise_loss = self._denoise_criterion(a,b)#(u_list[-1], u_list[-2])
            #print('loss',denoise_loss)#loss tensor(0.6566, device='cuda:0', grad_fn=<AddBackward0>)

            denoise_loss.backward()
            nn.utils.clip_grad_norm(self.denoise_net.parameters(), 5)
            self.denoise_optimizer.step()

        return enhancement_loss, denoise_loss, u_list


class DenoiseLossFunction(nn.Module):
    def __init__(self):
        super(DenoiseLossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()
        self.tv_loss = TVLoss()

    def forward(self, output, target):
        return 0.0000001 * self.l2_loss(output, target) + self.tv_loss(output)
    '''def forward(self, output,):
        return 0.0000001 * self.l2_loss(output[-1], output[-2]) + self.tv_loss(output)'''


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.l2_loss = nn.L1Loss()#nn.MSELoss()
        # self.smooth_loss = SmoothLoss()

    def forward(self, output, target):
        #print(output.shape,target.shape) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 96, 96])
        return self.l2_loss(output, target)
class LossFunctionorg(nn.Module):
    def __init__(self):
        super(LossFunctionorg, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()

    def forward(self, input, u_list, t_list):
        Fidelity_Loss = 0
        # Fidelity_Loss = Fidelity_Loss + self.l2_loss(output_list[i], input_list[i])
        # Smooth_Loss = 0
        # Smooth_Loss = Smooth_Loss + self.smooth_loss(input_list[i], output_list[i])
        i = input
        o = t_list[-1]
        # for i, o in zip(input_list, output_list):
        Fidelity_Loss = Fidelity_Loss + self.l2_loss(o, i)

        Smooth_Loss = 0
        # for i, o in zip(input_list, output_list):
        Smooth_Loss = Smooth_Loss + self.smooth_loss(i, o)
        # for d in d_list[:-1]:
        #   Smooth_Loss = Smooth_Loss + self.smooth_loss(input_list[0], d)
        # Alpha_Loss = 0
        # for a in alpha_list:
        #     Alpha_Loss = Alpha_Loss + self.smooth_loss(input_list[0], a)

        return 0.5 * Fidelity_Loss + Smooth_Loss


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 0.1

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(1, 3, input_im.shape[2], input_im.shape[3])
        return out

    def norm(self, tensor, p):
        return torch.mean(torch.pow(torch.abs(tensor), p))

    # output: output      input:input
    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        # print(self.input.shape)
        sigma_color = -1.0 / 2 * self.sigma * self.sigma
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * self.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p)
        pixel_grad2 = w2 * self.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p)
        pixel_grad3 = w3 * self.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p)
        pixel_grad4 = w4 * self.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p)
        pixel_grad5 = w5 * self.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p)
        pixel_grad6 = w6 * self.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p)
        pixel_grad7 = w7 * self.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p)
        pixel_grad8 = w8 * self.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p)
        pixel_grad9 = w9 * self.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p)
        pixel_grad10 = w10 * self.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p)
        pixel_grad11 = w11 * self.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p)
        pixel_grad12 = w12 * self.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p)
        pixel_grad13 = w13 * self.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p)
        pixel_grad14 = w14 * self.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p)
        pixel_grad15 = w15 * self.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p)
        pixel_grad16 = w16 * self.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p)
        pixel_grad17 = w17 * self.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p)
        pixel_grad18 = w18 * self.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p)
        pixel_grad19 = w19 * self.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p)
        pixel_grad20 = w20 * self.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p)
        pixel_grad21 = w21 * self.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p)
        pixel_grad22 = w22 * self.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p)
        pixel_grad23 = w23 * self.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p)
        pixel_grad24 = w24 * self.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term


class IlluLoss(nn.Module):
    def __init__(self):
        super(IlluLoss, self).__init__()

    def forward(self, input_I_low, input_im):
        input_gray = self.rgb_to_gray(input_im)
        low_gradient_x, low_gradient_y = self.compute_image_gradient(input_I_low)
        input_gradient_x, input_gradient_y = self.compute_image_gradient(input_gray)

        less_location_x = input_gradient_x < 0.01
        input_gradient_x = input_gradient_x.masked_fill_(less_location_x, 0.01)
        less_location_y = input_gradient_y < 0.01
        input_gradient_y = input_gradient_y.masked_fill_(less_location_y, 0.01)

        x_loss = torch.abs(torch.div(low_gradient_x, input_gradient_x))
        y_loss = torch.abs(torch.div(low_gradient_y, input_gradient_y))
        mut_loss = (x_loss + y_loss).mean()
        return mut_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def compute_image_gradient_o(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        grad_x = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        grad_y = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1])

        grad_min_x = torch.min(grad_x)
        grad_max_x = torch.max(grad_x)
        grad_norm_x = torch.div((grad_x - grad_min_x), (grad_max_x - grad_min_x + 0.0001))

        grad_min_y = torch.min(grad_y)
        grad_max_y = torch.max(grad_y)
        grad_norm_y = torch.div((grad_y - grad_min_y), (grad_max_y - grad_min_y + 0.0001))
        return grad_norm_x, grad_norm_y

    def compute_image_gradient(self, x):
        kernel_x = [[0, 0], [-1, 1]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

        kernel_y = [[0, -1], [0, 1]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

        weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

        grad_x = torch.abs(F.conv2d(x, weight_x, padding=1))
        grad_y = torch.abs(F.conv2d(x, weight_y, padding=1))

        grad_min_x = torch.min(grad_x)
        grad_max_x = torch.max(grad_x)
        grad_norm_x = torch.div((grad_x - grad_min_x), (grad_max_x - grad_min_x + 0.0001))

        grad_min_y = torch.min(grad_y)
        grad_max_y = torch.max(grad_y)
        grad_norm_y = torch.div((grad_y - grad_min_y), (grad_max_y - grad_min_y + 0.0001))
        return grad_norm_x, grad_norm_y

    def rgb_to_gray(self, x):
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        return gray
