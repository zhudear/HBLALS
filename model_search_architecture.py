import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import PRIMITIVES2
from genotypes import Genotype
#from network_usrnet import *
from scipy.io import loadmat
import os
from tools import utils_image as util
from losses import pytorch_ssim
#import lpips
import itertools
#from networkv import ResUNet #VedioHigherResolutionNet
#from default import _C as cfg2
import math
from losses.Perceptual_loss import *
#from network_usrnet import ResUNet as net
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

class MixedOp2(nn.Module):
    def __init__(self, C_in,C_out, stride):
        super(MixedOp2, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES2:
            #lambda C, kernel, dialtion, affine:
            #op = OPS[primitive](C_in, C_out)
            op = OPS[primitive](C_in, C_out)
            #op =nn.Sequential(op, nn.BatchNorm2d(C_out, affine=False))
            '''op = OPS[primitive](C_in,C_in)#3:kernel
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C_in, affine=False))'''
            self._ops.append(op)

    def forward(self, x, weights):
        '''for w, op in zip(weights, self._ops):
            print('MixedOp2',w)
            m=op(x)
            print(m.shape)'''
        return sum(w * op(x) for w, op in zip(weights, self._ops))
class MixedOp(nn.Module):
    def __init__(self, C_in, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            #lambda C, kernel, dialtion, affine:
            #op = OPS[primitive](C_in, C_out)
            op = OPS[primitive](C_in, C_in)
            #op =nn.Sequential(op, nn.BatchNorm2d(C_in, affine=False))
            '''op = OPS[primitive](C_in,C_in)#3:kernel
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C_in, affine=False))'''
            self._ops.append(op)

    def forward(self, x, weights):
        '''for w, op in zip(weights, self._ops):
            print('MixedOp',w.shape, op(x).shape)'''
        return sum(w * op(x) for w, op in zip(weights, self._ops))
class SearchBlock2(nn.Module):
    def __init__(self, channel):
        super(SearchBlock2, self).__init__()
        stride=1
        self.head = MixedOp2(channel[0], channel[1], stride)
        self.conv1=MixedOp2(channel[1], channel[1], stride)#OPS[op_names[0]](channel[0], channel[1])# kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv1_2=MixedOp2(channel[1], channel[1], stride)
        '''self.ce1=nn.Sequential(self.conv1,
                               nn.ReLU(inplace=True),
                               self.conv1_2)'''
        self.conv2 =MixedOp2(channel[1], channel[1], stride)# OPS[op_names[1]](channel[1], channel[2])
        self.conv2_2 = MixedOp2(channel[1], channel[1], stride)
        '''self.ce2 = nn.Sequential(self.conv2,
                                 nn.ReLU(inplace=True),
                                 self.conv2_2)'''
        self.down1=nn.Conv2d(channel[1], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)#nn.AvgPool2d(4,2,1)

        self.conv3 = MixedOp2(channel[2], channel[2],
                              stride)  # OPS[op_names[0]](channel[0], channel[1])# kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv3_2 = MixedOp2(channel[2], channel[2], stride)
        '''self.ce3 = nn.Sequential(self.conv3,
                                 nn.ReLU(inplace=True),
                                 self.conv3_2)'''
        self.conv4 = MixedOp2(channel[2], channel[2], stride)  # OPS[op_names[1]](channel[1], channel[2])
        self.conv4_2 = MixedOp2(channel[2], channel[2], stride)
        '''self.ce4 = nn.Sequential(self.conv4,
                                 nn.ReLU(inplace=True),
                                 self.conv4_2)'''
        self.down2 =nn.Conv2d(channel[2], channel[2], kernel_size=(2, 2), stride=(2, 2), bias=False)# nn.AvgPool2d(4, 2, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input,weights):
        #print('SearchBlock2',input.shape,weights[0].shape)#SearchBlock2 torch.Size([1, 3, 192, 192]) torch.Size([4])
        input = self.head(input, weights[0])
        x0=self.conv1(input, weights[1])
        x0=self.relu(x0)
        x0 = self.conv1_2(x0, weights[2])
        #x0 = self.ce1(input, weights[1], weights[2])
        x0=input+x0
        x1 = self.conv2(x0, weights[3])
        x1 = self.relu(x1)
        x1 = self.conv2_2(x1, weights[4])
        #x1=self.ce2(x0, weights[3], weights[4])
        x1=x1+x0
        x1=self.down1(x1)

        x2 = self.conv3(x1, weights[5])
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2, weights[6])

        #x2 = self.ce3(x1, weights[5], weights[6])
        x2 = x2 + x1

        x3 = self.conv4(x2, weights[7])
        x3 = self.relu(x3)
        x3 = self.conv4_2(x3, weights[8])
        #x3 = self.ce4(x2, weights[7], weights[8])
        x3 = x2 + x3
        x3 = self.down2(x3)
        '''input=self.head(input,weights[0])
        x1=self.conv1(input,weights[1])
        x1 = self.relu(x1)
        x1 = self.conv1_2(x1, weights[2])
        x1=x1+input
        x1=self.down1(x1)

        x2 = self.conv2(x1,weights[3])
        x2 = self.relu(x2)
        x2 = self.conv2_2(x2, weights[4])
        x2=x2+x1
        #x2 = x2 + x1
        x2 = self.down2(x2)'''
        #print('SearchBlock2',input.shape,x1.shape,x2.shape)
        #SearchBlock2 torch.Size([1, 32, 192, 192]) torch.Size([1, 64, 96, 96]) torch.Size([1, 64, 48, 48])
        return x3,x1#x2,x1
class SearchBlock3(nn.Module):
    def __init__(self, channel):
        super(SearchBlock3, self).__init__()
        stride=1
        self.up1 = nn.ConvTranspose2d(channel[0], channel[0], kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv5 =MixedOp2(channel[0], channel[0], stride)# OPS[op_names[0]](channel[0], channel[1])  # kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv5_2 =MixedOp2(channel[0], channel[0], stride)
        self.conv6 = MixedOp2(channel[0], channel[0],
                              stride)  # OPS[op_names[0]](channel[0], channel[1])  # kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv6_2 = MixedOp2(channel[0], channel[0], stride)
        # nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.ConvTranspose2d(channel[0], channel[1], kernel_size=(2, 2), stride=(2, 2),
                                      bias=False)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = MixedOp2(channel[1], channel[1], stride)
        self.conv7_2 = MixedOp2(channel[1], channel[1], stride)
        self.conv8 = MixedOp2(channel[1], channel[1], stride)
        self.conv8_2 = MixedOp2(channel[1], channel[1], stride)
        self.relu = nn.ReLU(inplace=True)
        self.tail=MixedOp2(channel[1], channel[2], stride)
    def forward(self, input,x1,x0,weights):
        io = input + x1
        #print('SearchBlock3', input.shape, x1.shape)
        #SearchBlock3 torch.Size([1, 64, 48, 48]) torch.Size([1, 64, 48, 48])
        io = self.up1(io)
        #print('SearchBlock3', io.shape)
        x = self.conv5(io,weights[0])
        x = self.relu(x)
        x = self.conv5_2(x,weights[1])
        #x = x + io
        x = x + io
        x2 = self.conv6(x, weights[2])
        x2 = self.relu(x2)
        x2 = self.conv6_2(x2, weights[3])
        x2=x2+x
        io2=x2+x0
        io2 = self.up2(io2)
        x = self.conv7(io2,weights[4])
        x = self.relu(x)
        x = self.conv7_2(x,weights[5])
        x3=x+io2
        x = self.conv8(x3, weights[6])
        x = self.relu(x)
        x = self.conv8_2(x, weights[7])
        x=x+x3
        x = self.tail(x,weights[8])
        #x = x + io2
        #print('SearchBlock3', input.shape, io.shape, io2.shape)
        return x
class SearchBlock(nn.Module):

    def __init__(self, channel):
        super(SearchBlock, self).__init__()

        self.channel = channel
        stride = 1
        self.cr1=MixedOp(self.channel, stride)
        self.crbn1=nn.BatchNorm2d(self.channel, affine=False)
        self.cr2=MixedOp(self.channel, stride)
        self.crbn2=nn.BatchNorm2d(self.channel, affine=False)
        self.cr3=MixedOp(self.channel, stride)
        self.crbn3=nn.BatchNorm2d(self.channel, affine=False)
        self.cr4 = MixedOp(self.channel, stride)
        self.crbn4 = nn.BatchNorm2d(self.channel, affine=False)

        self.ci1 = MixedOp(self.channel, stride)
        self.cibn1=nn.BatchNorm2d(self.channel, affine=False)
        self.ci2 = MixedOp(self.channel, stride)
        self.cibn2 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci3 = MixedOp(self.channel, stride)
        self.cibn3 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci4 = MixedOp(self.channel, stride)
        self.cibn4 = nn.BatchNorm2d(self.channel, affine=False)
        self.sigmoid=nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputr,inputi, weights):
        #print(len(weights)) 4
        reflectance=self.cr1(inputr, weights[0])
        reflectance=self.crbn1(reflectance)
        reflectance=self.relu(reflectance)

        reflectance = self.cr2(reflectance, weights[1])
        reflectance = self.crbn2(reflectance)
        reflectance = self.relu(reflectance)

        reflectance = self.cr3(reflectance, weights[2])
        reflectance = self.crbn3(reflectance)
        reflectance = self.relu(reflectance)

        reflectance = self.cr4(reflectance, weights[3])
        reflectance = self.crbn4(reflectance)

        reflectance = reflectance + inputr
        reflectance = self.relu(reflectance)

        illumination = self.ci1(inputi, weights[4])
        illumination = self.cibn1(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci2(illumination, weights[5])
        illumination = self.cibn2(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci3(illumination, weights[6])
        illumination = self.cibn3(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci4(illumination, weights[7])
        illumination = self.cibn4(illumination)

        illumination = illumination + inputi
        illumination = self.relu(illumination)

        ytemp = self.sigmoid(illumination)
        zero = torch.zeros_like(ytemp)
        zero += 0.01
        ytemp = torch.max(ytemp, zero)
        ytemp = torch.div(reflectance, ytemp)
        return ytemp
class SearchBlock1(nn.Module):

    def __init__(self, channel):
        super(SearchBlock, self).__init__()

        self.channel = channel
        stride = 1
        self.cr= MixedOp(self.channel, stride)#self.dc)
        self.ci = MixedOp(self.channel,stride)# self.rc)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input, weights):
        reflectance=self.cr(input, weights[0])
        illumination=self.ci(input, weights[1])
        ytemp = self.sigmoid(illumination)
        zero = torch.zeros_like(ytemp)
        zero += 0.01
        ytemp = torch.max(ytemp, zero)
        ytemp = torch.div(reflectance, ytemp)
        return ytemp
class SearchBlocksub(nn.Module):

    def __init__(self, channel):
        super(SearchBlocksub, self).__init__()

        self.channel = channel

        stride = 1
        self.cr1=MixedOp(self.channel, stride)
        #self.crbn1=nn.BatchNorm2d(self.channel, affine=False)
        self.cr2=MixedOp(self.channel, stride)
        #self.crbn2=nn.BatchNorm2d(self.channel, affine=False)
        self.cr3=MixedOp(self.channel, stride)
        #self.crbn3=nn.BatchNorm2d(self.channel, affine=False)
        self.cr4 = MixedOp(self.channel, stride)
        #self.crbn4 = nn.BatchNorm2d(self.channel, affine=False)

        self.ci1 = MixedOp(self.channel, stride)
        #self.cibn1=nn.BatchNorm2d(self.channel, affine=False)
        self.ci2 = MixedOp(self.channel, stride)
        #self.cibn2 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci3 = MixedOp(self.channel, stride)
        #self.cibn3 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci4 = MixedOp(self.channel, stride)
        #self.cibn4 = nn.BatchNorm2d(self.channel, affine=False)
        self.sigmoid=nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputr,inputi, weights):
        #print(len(weights)) 4
        reflectance=self.cr1(inputr, weights[0])
        #reflectance=self.crbn1(reflectance)
        reflectance=self.relu(reflectance)

        reflectance = self.cr2(reflectance, weights[1])
        #reflectance = self.crbn2(reflectance)
        reflectance = self.relu(reflectance)

        reflectance = self.cr3(reflectance, weights[2])
        #reflectance = self.crbn3(reflectance)
        reflectance = self.relu(reflectance)

        reflectance = self.cr4(reflectance, weights[3])
        #reflectance = self.crbn4(reflectance)

        reflectance = reflectance + inputr
        reflectance = self.relu(reflectance)

        illumination = self.ci1(inputi, weights[4])
        #illumination = self.cibn1(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci2(illumination, weights[5])
        #illumination = self.cibn2(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci3(illumination, weights[6])
        #illumination = self.cibn3(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci4(illumination, weights[7])
        #illumination = self.cibn4(illumination)

        illumination = illumination + inputi
        illumination = self.relu(illumination)

        ytemp = self.sigmoid(illumination)
        ytemp=torch.clamp(ytemp,0.01,1)
        '''zero = torch.zeros_like(ytemp)
        zero += 0.01
        ytemp = torch.max(ytemp, zero)'''
        ytemp = torch.sub(reflectance, ytemp)
        return ytemp
class SearchBlockdiv(nn.Module):

    def __init__(self, channel):
        super(SearchBlockdiv, self).__init__()
        #print('only one bn')

        self.channel = channel

        stride = 1
        self.cr1=MixedOp(self.channel, stride)
        #self.crbn1=nn.BatchNorm2d(self.channel, affine=False)
        self.cr2=MixedOp(self.channel, stride)
        #self.crbn2=nn.BatchNorm2d(self.channel, affine=False)
        self.cr3=MixedOp(self.channel, stride)
        #self.crbn3=nn.BatchNorm2d(self.channel, affine=False)
        self.cr4 = MixedOp(self.channel, stride)
        #self.crbn4 = nn.BatchNorm2d(self.channel, affine=False)

        self.ci1 = MixedOp(self.channel, stride)
        #self.cibn1=nn.BatchNorm2d(self.channel, affine=False)
        self.ci2 = MixedOp(self.channel, stride)
        #self.cibn2 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci3 = MixedOp(self.channel, stride)
        #self.cibn3 = nn.BatchNorm2d(self.channel, affine=False)
        self.ci4 = MixedOp(self.channel, stride)
        #self.cibn4 = nn.BatchNorm2d(self.channel, affine=False)
        self.sigmoid=nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputr,inputi, weights):
        #print(len(weights)) 4
        reflectance=self.cr1(inputr, weights[0])
        #reflectance=self.crbn1(reflectance)
        reflectance=self.relu(reflectance)

        reflectance = self.cr2(reflectance, weights[1])
        #reflectance = self.crbn2(reflectance)
        reflectance = self.relu(reflectance)

        reflectance = self.cr3(reflectance, weights[2])
        #reflectance = self.crbn3(reflectance)
        reflectance = self.relu(reflectance)

        reflectance = self.cr4(reflectance, weights[3])
        #reflectance = self.crbn4(reflectance)

        reflectance = reflectance + inputr
        reflectance = self.relu(reflectance)

        illumination = self.ci1(inputi, weights[4])
        #illumination = self.cibn1(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci2(illumination, weights[5])
        #illumination = self.cibn2(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci3(illumination, weights[6])
        #illumination = self.cibn3(illumination)
        illumination = self.relu(illumination)

        illumination = self.ci4(illumination, weights[7])
        #illumination = self.cibn4(illumination)

        illumination = illumination + inputi
        illumination = self.relu(illumination)

        ytemp = self.sigmoid(illumination)
        ytemp=torch.clamp(ytemp,0.01,1)
        '''zero = torch.zeros_like(ytemp)
        zero += 0.01
        ytemp = torch.max(ytemp, zero)'''
        ytemp = torch.div(reflectance, ytemp)
        return ytemp
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
#HyPaLoss
class HyPaLoss(nn.Module):
    def __init__(self, layers, channel):
        super(HyPaLoss, self).__init__()
        self.nrm_nums = layers
        self.channel = channel
        self.stem = conv_layer(3, self.channel, 3)
        self.nrms = nn.ModuleList()
        for i in range(self.nrm_nums):
            self.nrms.append(SearchBlock(self.channel))
        self.activate = nn.Sequential(conv_layer(self.channel, 4, 3))
        self.pool=nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid=nn.Sigmoid()
    def forward(self, input, weights):
        #print('DenoiseNetwork',input.shape,b.shape)
        #DenoiseNetwork torch.Size([1, 3, 128, 128]) torch.Size([1, 1, 16384, 16384])
        x =input

        feat = self.stem(x)
        for i in range(self.nrm_nums):
            feat = self.nrms[i](feat, weights[i])
            #feat = self.nrms[i](feat, weights[0])
        n = self.activate(feat)
        n=self.pool(n)
        n=self.sigmoid(n)
        #print('HyPaLoss', input.shape,n.shape,n)#torch.Size([1, 4, 1, 1])
        #HyPaLoss torch.Size([1, 3, 16, 16]) torch.Size([1, 3, 16, 16])
        #a list, the weight of each loss function
        return n
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        in_channels=3
        mid_channels=16
        out_channels=32
        self.encode=SearchBlock2([in_channels,mid_channels,out_channels])
    def forward(self, input,weights):
        #print('enc',weights[0].shape)
        '''h, w = input.size()[-2:]
        paddingBottom = int(np.ceil(h / 4) * 4 - h)#8
        paddingRight = int(np.ceil(w / 4) * 4 - w)
        input = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(input)'''
        return self.encode(input,weights[0])
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_channels = 32
        mid_channels = 16
        out_channels = 3
        self.decode=SearchBlock3([in_channels,mid_channels,out_channels])
    def forward(self, input,x1,x0,weights):
        return self.decode(input,x1,x0,weights[0])
class DenoiseNetworksub(nn.Module):

    def __init__(self, layers, channel,operator):
        super(DenoiseNetworksub, self).__init__()

        self.nrm_nums = layers
        self.channel = channel
        #self.stem = conv_layer(4, self.channel, 3)
        self.operator=operator
        self.nrms = nn.ModuleList()
        for i in range(self.nrm_nums):
            self.nrms.append(SearchBlocksub(self.channel))

        #self.activate = nn.Sequential(conv_layer(self.channel, 3, 3))

    def forward(self, input, weights):
        #print('deno', weights[0].shape, weights[1].shape)#deno torch.Size([4, 7]) torch.Size([4, 7])
        x = input
        # feat = self.stem(x)
        for i in range(self.nrm_nums):
            x = self.nrms[i](x, x,weights[i])
        return x


class DenoiseNetworkdiv(nn.Module):

    def __init__(self, layers, channel, operator):
        super(DenoiseNetworkdiv, self).__init__()

        self.nrm_nums = layers
        self.channel = channel
        # self.stem = conv_layer(4, self.channel, 3)
        self.operator = operator
        self.nrms = nn.ModuleList()
        for i in range(self.nrm_nums):
            self.nrms.append(SearchBlockdiv(self.channel))

        # self.activate = nn.Sequential(conv_layer(self.channel, 3, 3))

    def forward(self, input, weights):
        # print('deno', weights[0].shape, weights[1].shape)#deno torch.Size([4, 7]) torch.Size([4, 7])
        x = input
        # feat = self.stem(x)
        for i in range(self.nrm_nums):
            x = self.nrms[i](x, x, weights[i])
        return x
class DenoiseNetwork(nn.Module):

    def __init__(self, layers, channel,operator):
        super(DenoiseNetwork, self).__init__()

        self.nrm_nums = layers
        self.channel = channel
        #self.stem = conv_layer(4, self.channel, 3)
        self.operator=operator
        self.nrms = nn.ModuleList()
        if self.operator == '-':
            # print('self.operator -')
            for i in range(self.nrm_nums):
                self.nrms.append(SearchBlocksub(self.channel))
        else:
            #print('self.operator div')
            for i in range(self.nrm_nums):
                self.nrms.append(SearchBlockdiv(self.channel))
        #self.activate = nn.Sequential(conv_layer(self.channel, 3, 3))

    def forward(self, input, weights):
        #print('deno', weights[0].shape, weights[1].shape)#deno torch.Size([4, 7]) torch.Size([4, 7])
        x = input
        # feat = self.stem(x)
        for i in range(self.nrm_nums):
            x = self.nrms[i](x, x,weights[i])
        return x
class DenoiseNetworkorg(nn.Module):

    def __init__(self, layers, channel):
        super(DenoiseNetwork, self).__init__()

        self.nrm_nums = layers
        self.channel = channel
        self.layer = nn.Sequential(nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                   nn.BatchNorm2d(self.channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                   nn.BatchNorm2d(self.channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                   )
        self.trasition = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        #self.stem = conv_layer(4, self.channel, 3)
        self.nrms = nn.ModuleList()
        for i in range(self.nrm_nums):
            self.nrms.append(SearchBlock(self.channel))
        #self.activate = nn.Sequential(conv_layer(self.channel, 3, 3))

    def forward(self, input, weights):
        #print('deno', weights[0].shape, weights[1].shape)#deno torch.Size([4, 7]) torch.Size([4, 7])
        input = self.layer(input)
        x = self.trasition(input)

        x = self.nrms[0](input, x,weights[0])
        # feat = self.stem(x)
        for i in range(self.nrm_nums - 1):
            x = self.nrms[i + 1](x, x,weights[i+1])
        return x

class Encodero(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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
        super(Decoder, self).__init__()
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


class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        in_channels=3
        mid_channels=16
        out_channels=32
        self.ce1=nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.ce2=nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        '''self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))'''

    def forward(self, input):
        #print('DenoiseNetwork',input.shape,b.shape)
        #DenoiseNetwork torch.Size([1, 3, 128, 128]) torch.Size([1, 1, 16384, 16384])
        x0 =self.ce1(input)
        x1 = self.ce2(x0)

        #print('HyPaLoss', input.shape,n.shape,n)#torch.Size([1, 4, 1, 1])
        #HyPaLoss torch.Size([1, 3, 16, 16]) torch.Size([1, 3, 16, 16])
        #a list, the weight of each loss function
        return x1,x0
class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_channels = 32
        mid_channels = 16
        out_channels = 3
        self.cd1 = nn.Sequential(
            nn.Conv2d(in_channels*2, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.cd2 = nn.Sequential(
            nn.Conv2d(mid_channels*2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        '''self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))'''

    def forward(self, input,x1,x0):
        #print(input.shape,x1.shape,x0.shape)
        # print('DenoiseNetwork',input.shape,b.shape)
        # DenoiseNetwork torch.Size([1, 3, 128, 128]) torch.Size([1, 1, 16384, 16384])
        x=torch.cat([input, x1], dim=1)
        x = self.cd1(x)
        x = torch.cat([x, x0], dim=1)
        x = self.cd2(x)

        # print('HyPaLoss', input.shape,n.shape,n)#torch.Size([1, 4, 1, 1])
        # HyPaLoss torch.Size([1, 3, 16, 16]) torch.Size([1, 3, 16, 16])
        # a list, the weight of each loss function
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

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
class Gray_world_Loss(nn.Module):
    def __init__(self):
        super(Gray_world_Loss, self).__init__()
        #self.l1=LossFunctionl1()
        self.l1 = nn.L1Loss()  # nn.MSELoss()
    def forward(self, target):
        #print('target',target)
        '''B, G, R = np.double(orgImg[:, 0,:,:]), np.double(orgImg[:, 1,:,:]), np.double(orgImg[:, 2,:,:])
        B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)'''

        #tB, tG, tR = np.double(target[:, 0, :, :]), np.double(target[:, 1, :, :]), np.double(target[:, 2, :, :])
        tB, tG, tR = target[:,  0, :, :],target[:, 1, :, :], target[:, 2, :, :]
        tB_ave, tG_ave, tR_ave = torch.mean(tB), torch.mean(tG), torch.mean(tR)
        #tensor(-3.7253e-09, device='cuda:1', grad_fn=<MeanBackward0>) tensor(3.7253e-09, device='cuda:1', grad_fn=<MeanBackward0>) tensor(-1.1176e-08, device='cuda:1', grad_fn=<MeanBackward0>)
        #loss_Gray=self.l1(tB_ave, tG_ave)+self.l1(tG_ave, tR_ave)+self.l1(tB_ave, tR_ave)
        loss_Gray=torch.abs(tB_ave-tG_ave)+torch.abs(tG_ave- tR_ave)+torch.abs(tB_ave-tR_ave)
        #print('l~~~~~~~~~~~~~l',tB_ave, tG_ave, tR_ave,loss_Gray)

        return loss_Gray#100*
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
#import VIF_loss
class VIFloss(nn.Module):
    def __init__(self):
        super(VIFloss,self).__init__()
    def forward(self,ref, dist):
        #0-255
        loss=VIF_loss.vifp_mscale(ref, dist)
        return 2-loss#0=255 range max is 8
#from Laplaian import LapLoss
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
class Network(nn.Module):

    def __init__(self,lw,operator):
        super(Network, self).__init__()
        self.lw=lw
        self.nrm_nums = 7
        self.operator = operator
        '''self.hloss_nums = 1
        self.denoise_channel = 64#12

        self.sf = 4
        self.sigma = torch.tensor(0).float().view([1, 1, 1, 1]).cuda()

        self.n = 4#4'''
        #lw = [1.38, 1.28, 0.82, 1.23, 0, 0.74, 0, 0]
        #self._criterion = LossFunction()
        self.l1_criterion=LossFunctionl1()
        self.l2_criterion=LossFunctionl2()
        self.color_criterion = LossFunctionlcolor()
        self.ssim_criterion = pytorch_ssim.SSIM(window_size=11)#LossFunctionssim()
        self.vgg_criterion=LossFunctionvgg2()
        #self.vgg_criterion=LossFunctionvgg2()
        #self.tv_criterion=TVLoss()
        #self.smoothnl1=LossFunctionsmoothl1()
        '''self.loss_fn_alex = lpips.LPIPS(net='alex')
        for k in self.loss_fn_alex.parameters():
            k.requires_grad = False'''
        # self.niqe=LossNIQE()

        # self.vif=
        # self.tv = TVLoss()
        self.lap = LapLoss()


        #self.discr_criterion=LossDiscrim()

        #self.classer_criterion=LossClasser()
        #self._initialize_alphas()
        self._initialize_alphas()

        # self.vgg_loss = PerceptualLoss()
        # self.vgg_loss.cuda()
        # self.vgg = load_vgg16("./model")
        # self.vgg.eval()
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

        '''self.unet_model =ResUNet(cfg2, out_nc=3, nc=[16, 32] , nb=2,
                                      act_mode="R", downsample_mode="strideconv", upsample_mode="convtranspose")'''
        self.e=Encoder()

        self.p=DenoiseNetwork(self.nrm_nums,32,self.operator)#sub
        self.d =Decoder()
        #self.sigmoid = nn.Sigmoid()
        #VedioHigherResolutionNet(cfg2)
        '''self.hyper_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_6 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_7 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_8 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)'''
        #self.hyper_9 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.hyper_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_6 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_7 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_8 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)


        '''self.hyper_7 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_8 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
        self.hyper_9 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)'''
        # 初始化
        # self.hyper_1.data.fill_(0.509130)
        # self.hyper_2.data.fill_(3.364720)
        # self.hyper_3.data.fill_(0.001258)#0.1
        # self.hyper_4.data.fill_(7.304316)#10
        # self.hyper_5.data.fill_(2.460482)
        # self.hyper_6.data.fill_(2.540187)
        # self.hyper_7.data.fill_(0)
        # self.hyper_8.data.fill_(0)
        # self.hyper_9.data.fill_(0.672123)
#epoch 60:  0.591911 0.000000 0.000000 0.000702 5.892449 0.000000 3.562198 2.124476 1.323368
        #print('lw',lw)
        self.hyper_1.data.fill_(lw[0])
        self.hyper_2.data.fill_(lw[1])
        self.hyper_3.data.fill_(lw[2])  # 0.1
        self.hyper_4.data.fill_(lw[3])  # 10
        self.hyper_5.data.fill_(lw[4])
        self.hyper_6.data.fill_(lw[5])
        self.hyper_7.data.fill_(lw[6])
        self.hyper_8.data.fill_(lw[7])

        '''self.hyper_7.data.fill_(lw[6])
        self.hyper_8.data.fill_(lw[7])'''
        #self.hyper_9.data.fill_(lw[8])

        #self.hyper_6.data.fill_(1)

    def _initialize_alphas(self): # sum(1 for i in range(self.illu_layers))
        k_denoise = 8#2  # sum(1 for i in range(self.alph_layers))
        num_ops = len(PRIMITIVES)
        self.alphas_denoises = []
        for i in range(self.nrm_nums):
            a = torch.tensor(1e-3 * torch.randn(k_denoise, num_ops).cuda(), requires_grad=True)
            self.alphas_denoises.append(a)
        num_ops2 = len(PRIMITIVES2)
        self.alphas_e = []
        a = torch.tensor(1e-3 * torch.randn(9, num_ops2).cuda(), requires_grad=True)
        self.alphas_e.append(a)
        self.alphas_d = []
        a = torch.tensor(1e-3 * torch.randn(9, num_ops2).cuda(), requires_grad=True)
        self.alphas_d.append(a)
        #print('_initialize_alphas',a.shape,self.alphas_denoises[0].shape)#torch.Size([9, 4]) torch.Size([8, 9])
        '''vgg feature layers'''
        '''self.vgg1=torch.tensor(1e-3 * torch.randn(1, 4).cuda(), requires_grad=True)
        self.vgg2=torch.tensor(1e-3 * torch.randn(1, 4).cuda(), requires_grad=True)
        self.vgg3=torch.tensor(1e-3 * torch.randn(1, 8).cuda(), requires_grad=True)
        self.vgg4=torch.tensor(1e-3 * torch.randn(1, 8).cuda(), requires_grad=True)'''
    def new(self):
        model_new = Network(self.lw,self.operator).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            #print('$',x)
            x.data.copy_(y.data)
        return model_new
    def hyper_parameters(self):
        #print('get hyper_parameters')#,self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4)
        return [self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4,self.hyper_5
            ,self.hyper_6, self.hyper_7, self.hyper_8,self.vgg1,self.vgg2,self.vgg3,self.vgg4]# ,self.hyper_5]

    def new_hyper(self):
        model_new = Network(self.lw,self.operator).cuda()
        for x, y in zip(model_new.hyper_parameters(), self.hyper_parameters()):
            #print(x,y,'%')
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):#

        denoise_weights = []
        for i in range(self.nrm_nums):
            denoise_weights.append(F.softmax(self.alphas_denoises[i], dim=-1))
        e_weights = []
        e_weights.append(F.softmax(self.alphas_e[0], dim=-1))
        d_weights = []
        d_weights.append(F.softmax(self.alphas_d[0], dim=-1))
        x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x)

        #print('self.e,p,d')
        #print('self.e,p,d',denoise_weights[0].shape,e_weights[0].shape)#self.e,p,d torch.Size([4, 7]) torch.Size([2, 4])
        x1,x0=self.e(x,e_weights)#enc torch.Size([2, 4])
        x2=self.p(x1,denoise_weights) #x1
        x3=self.d(x2,x1,x0,d_weights)
        #x3=self.sigmoid(x3)
        #print('forword',x.shape,x0.shape,x1.shape,x2.shape,x3.shape)#reflection,Illimination1,
        #forword torch.Size([1, 3, 192, 192]) torch.Size([1, 32, 192, 192]) torch.Size([1, 32, 192, 192]) torch.Size([1, 3, 192, 192])
        #k =self.kernel# torch.tensor(1).float().repeat([1, 1, 25, 25]).cuda()
        #print('forword',x.shape)
        x3 = pad_tensor_back(x3, pad_left, pad_right, pad_top, pad_bottom)
        return x3#,reflection,Illimination1

    def _loss2(self, input, target, vgg_model, discriminator, type):
        # print('_loss')

        output = self(input)  # ,reflection,Illimination1
        '''hloss_weights = []
        for i in range(self.hloss_nums):
            hloss_weights.append(F.softmax(self.alphas_loss[i], dim=-1))
        hloss = self.hloss(input,hloss_weights)'''
        # print(input.shape, target.shape, output.shape)
        # torch.Size([1, 3, 24, 24]) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 48, 48])
        # print(len(hloss)) #16
        finalloss = []
        if (type == 1):  # True:# False:# train
            # 0,0,0,5.6,0.87,0.9,0.6,0,0.87,0
            l1loss = self.hyper_1 * self.l1_criterion(output, target)
            l2loss = self.hyper_2 * self.l2_criterion(output, target)
            colorloss = self.hyper_3 * self.color_criterion(output, target)
            ssimloss = self.hyper_4 * (1 - self.ssim_criterion(output, target))
            # vggloss = self.hyper_5 * self.vgg_criterion(output, target, vgg_model)

            vggloss = self.hyper_5 * self.vgg_criterion(output, target, vgg_model)

            lpipsloss = self.hyper_6 * self.loss_fn_alex(output, target)
            tvloss = self.hyper_8 * self.tv(output)
            laploss = self.hyper_9 * self.lap(output)  # , target)
            # self.smoothnl1(output, target)
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

            '''target = target * 255
            # print(img)
            # (R*299 + G*587 + B*114 + 500) / 1000
            # print(img.shape,img)#torch.Size([1, 3, 400, 600])
            target[:, 0:1, :, :] = target[:, 0:1, :, :] * 299
            target[:, 1:2, :, :] = target[:, 1:2, :, :] * 587
            target[:, 2:3, :, :] = target[:, 2:3, :, :] * 114
            target = torch.sum(target, dim=1)  # /1000
            # print(img.shape, target.shape)#torch.Size([1, 500, 333]) torch.Size([1, 500, 333])

            target = target / 1000'''
            vifloss = self.hyper_10 * self.vif(img, target)
            deloss = self.hyper_7 * self.discrete_entropy(img)

            finalloss.append(l1loss + l2loss + colorloss + ssimloss + vggloss +
                             lpipsloss + deloss + tvloss + laploss + vifloss)  # +niqeloss  +gwloss05+lploss08)#+lossg)
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
            '''finalloss.append(gwloss05)
            finalloss.append(lploss08)'''
            # finalloss.append(lossg)
            # finalloss.append(lpips)
        else:  # vail
            # loss = 0
            loss = self.classer_criterion(output, None, discriminator)
            # loss = self.loss_fn_alex(output, target)  0905
            # finalloss.append(loss)
            finalloss.append(loss[0])
            finalloss.append(loss[1])
        # +self._criterion(output[i*3+2], target)
        # loss = self._criterion(output[-1], target)
        '''print('############')
        for v in torch.autograd.grad(finalloss[0], self.net_parameters()):
            print(v)'''
        return finalloss
    def _loss(self, input, target, vgg_model,loss_fn_alex, discriminator, type):
        # print('_loss')

        output = self(input)  # ,reflection,Illimination1
        '''hloss_weights = []
        for i in range(self.hloss_nums):
            hloss_weights.append(F.softmax(self.alphas_loss[i], dim=-1))
        hloss = self.hloss(input,hloss_weights)'''
        # print(input.shape, target.shape, output.shape)
        # torch.Size([1, 3, 24, 24]) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 48, 48])
        # print(len(hloss)) #16
        finalloss = []
        if (type == 1):  # True:# False:# train
            #0,0,0,5.6,0.87,0.9,0.6,0,0.87,0
            #LOL lw = [1.38, 1.28, 0.82, 1.23, 0, 0.74, 0, 0]
            #underwater lw=[1.11, 1.09,1.40, 1.90, 1.21, 1.09,0,0]
            l1loss = self.hyper_1 * self.l1_criterion(output, target)
            l2loss = self.hyper_2 * self.l2_criterion(output, target)
            colorloss = self.hyper_3 * self.color_criterion(output, target)
            ssimloss = self.hyper_4 * 10*(1 - self.ssim_criterion(output, target))
            #vggloss = self.hyper_5 * self.vgg_criterion(output, target, vgg_model)

            vggloss =  self.hyper_5 * self.vgg_criterion(output, target, vgg_model)

            lpipsloss = self.hyper_6 *  loss_fn_alex(output, target)
            tvloss =0# self.hyper_8 * self.tv(output)
            laploss =0# self.hyper_8 * self.lap(output)#, target)
            #self.smoothnl1(output, target)

            '''target = target * 255
            # print(img)
            # (R*299 + G*587 + B*114 + 500) / 1000
            # print(img.shape,img)#torch.Size([1, 3, 400, 600])
            target[:, 0:1, :, :] = target[:, 0:1, :, :] * 299
            target[:, 1:2, :, :] = target[:, 1:2, :, :] * 587
            target[:, 2:3, :, :] = target[:, 2:3, :, :] * 114
            target = torch.sum(target, dim=1)  # /1000
            # print(img.shape, target.shape)#torch.Size([1, 500, 333]) torch.Size([1, 500, 333])
            
            target = target / 1000'''
            # vifloss =  self.hyper_10 * self.vif(img, target)
            # deloss = self.hyper_7 * self.discrete_entropy(img)

            finalloss.append(l1loss + l2loss + colorloss + ssimloss + vggloss +
                             lpipsloss  + tvloss + laploss )  # +niqeloss  +gwloss05+lploss08)#+lossg)
            finalloss.append(l1loss)
            finalloss.append(l2loss)
            finalloss.append(colorloss)
            finalloss.append(ssimloss)
            finalloss.append(vggloss)

            finalloss.append(lpipsloss)

            finalloss.append(tvloss)
            finalloss.append(laploss)

            '''finalloss.append(gwloss05)
            finalloss.append(lploss08)'''
            # finalloss.append(lossg)
            # finalloss.append(lpips)
        else:  # vail
            # loss = 0
            loss = self.classer_criterion(output, None, discriminator)
            # loss = self.loss_fn_alex(output, target)  0905
            # finalloss.append(loss)
            finalloss.append(loss[0])
            finalloss.append(loss[1])
        # +self._criterion(output[i*3+2], target)
        # loss = self._criterion(output[-1], target)
        '''print('############')
        for v in torch.autograd.grad(finalloss[0], self.net_parameters()):
            print(v)'''
        return finalloss

    def _loss3(self, input, target,vgg_model,discriminator,type):
        #print('_loss')

        output= self(input)#,reflection,Illimination1
        '''hloss_weights = []
        for i in range(self.hloss_nums):
            hloss_weights.append(F.softmax(self.alphas_loss[i], dim=-1))
        hloss = self.hloss(input,hloss_weights)'''
        #print(input.shape, target.shape, output.shape)
        #torch.Size([1, 3, 24, 24]) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 48, 48])
        #print(len(hloss)) #16
        finalloss=[]
        if (type==1):# True:# False:# train
            #lossg = self.hyper_9 * self.gradient(output)
            if(self.hyper_1==0):
                l1loss=0
            else:
                l1loss = self.hyper_1 * self.l1_criterion(output, target)
            if (self.hyper_2 == 0):
                l2loss = 0
            else:
                l2loss = self.hyper_2 * self.l2_criterion(
                output, target)
            if (self.hyper_3 == 0):
                colorloss = 0
            else:
                colorloss=self.hyper_3 *self.color_criterion(output, target)
            if (self.hyper_4 == 0):
                ssimloss = 0
            else:
                ssimloss =self.hyper_4 * (1 - self.ssim_criterion(output, target))
            #output_layer_list
            #vggloss = self.hyper_5 * self.vgg_criterion(vgg1, vgg2, vgg3,vgg4,
            #    output, target,input,vgg_model)
            if (self.hyper_5 == 0):
                vggloss = 0
            else:
                vggloss = self.hyper_5 * self.vgg_criterion(output, target, vgg_model)
            if (self.hyper_6 == 0):
                lpipsloss = 0
            else:
                lpipsloss= self.loss_fn_alex(output, target)
            if (self.hyper_8 == 0):
                tvloss = 0
            else:
                tvloss = self.hyper_8 * self.tv(output)
            if (self.hyper_9 == 0):
                laploss = 0
            else:
                laploss = self.hyper_9 * self.lap(output, target)
            img=None
            if (self.hyper_7 == 0):
                deloss = 0
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
                deloss = self.hyper_7 * self.discrete_entropy(img)
            if (self.hyper_10 == 0):
                vifloss = 0
            elif self.hyper_7 != 0:
                target = target * 255
                # print(img)
                # (R*299 + G*587 + B*114 + 500) / 1000
                # print(img.shape,img)#torch.Size([1, 3, 400, 600])
                target[:, 0:1, :, :] = target[:, 0:1, :, :] * 299
                target[:, 1:2, :, :] = target[:, 1:2, :, :] * 587
                target[:, 2:3, :, :] = target[:, 2:3, :, :] * 114
                target = torch.sum(target, dim=1)  # /1000
                # print(img.shape, target.shape)#torch.Size([1, 500, 333]) torch.Size([1, 500, 333])
                target = target / 1000
                vifloss = self.hyper_10 * self.vif(img, target)
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

            #z=self.tv_criterion(x)
            #lpips=20*self.loss_fn_alex(output,target)
            #print(l1loss.item(),l2loss.item(),colorloss.item(),ssimloss.item(),vggloss.item(),smoothl1loss.item(),lploss05.item(),lploss08.item(),lossg.item())
            finalloss.append(l1loss + l2loss + colorloss + ssimloss + vggloss +
                             lpipsloss + deloss + tvloss + laploss + vifloss)  # +niqeloss  +gwloss05+lploss08)#+lossg)
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
            '''finalloss.append(gwloss05)
            finalloss.append(lploss08)'''
            #finalloss.append(lossg)
            #finalloss.append(lpips)
        else:#vail
            #loss = 0
            loss=self.classer_criterion(output,None,discriminator)
            #loss = self.loss_fn_alex(output, target)  0905
            #finalloss.append(loss)
            finalloss.append(loss[0])
            finalloss.append(loss[1])
        #+self._criterion(output[i*3+2], target)
        #loss = self._criterion(output[-1], target)
        '''print('############')
        for v in torch.autograd.grad(finalloss[0], self.net_parameters()):
            print(v)'''
        return finalloss

    '''def get_alpha_beta(self):
        return self.soft_plus(self.alpha_beta)'''

    def net_named_parameters(self):
        return itertools.chain(self.e.named_parameters(), self.p.named_parameters(), self.d.named_parameters())
    def net_parameters(self):
        ''' print('netp_parameters')
        print(self.p.parameters())
        print('neth_parameters')
        print(self.h.parameters())'''
        #print(self.unet_model.parameters())
        #return itertools.chain(self.p.parameters(),self.h.parameters())
        return itertools.chain(self.e.parameters(),self.p.parameters(),self.d.parameters())#self.unet_model.parameters()
        #return itertools.chain(self.e.named_parameters(),self.p.named_parameters(),self.d.named_parameters())#self.unet_model.parameters()

    def arch_parameters(self):
        #print('arch_parameterspppp')
        #print(len(self.alphas_denoises),len(self.alphas_e),len(self.alphas_d))#5 1 1
        return [v for v in self.alphas_denoises] + [v for v in self.alphas_e]+[v for v in self.alphas_d]
    def hloss_arch_parameters(self):
        return [self.alphas_loss[0]]

    def enhance_arch_parameters(self):
        return [self.alphas_enhances[0]]

    def denoise_arch_parameters(self):
        return [self.alphas_denoises[0]]

    def enhance_net_parameters(self):
        return self.enhance_net.parameters()

    def denoise_net_parameters(self):
        return self.denoise_net.parameters()
    def genotype(self, i, task=''):
        def _parse(weights, layers,task):
            gene = []
            for i in range(layers):
                W = weights[i].copy()
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k
                if task == 'denoise':
                    gene.append((PRIMITIVES[k_best], i))
                else:
                    gene.append((PRIMITIVES2[k_best], i))
            return gene

        if task == 'enhance':
            gene = _parse(F.softmax(self.alphas_enhances[i], dim=-1).data.cpu().numpy(), 8,task)
        elif task == 'hloss':
            gene = _parse(F.softmax(self.alphas_loss[i], dim=-1).data.cpu().numpy(), 8,task)
        elif task == 'denoise':#alphas_loss
            gene = _parse(F.softmax(self.alphas_denoises[i], dim=-1).data.cpu().numpy(), 8,task)
        elif task == 'e':#alphas_loss
            gene = _parse(F.softmax(self.alphas_e[0], dim=-1).data.cpu().numpy(), 9,task)
        elif task == 'd':#alphas_loss
            gene = _parse(F.softmax(self.alphas_d[0], dim=-1).data.cpu().numpy(), 9,task)
        genotype = Genotype(
            normal=gene, normal_concat=None,
            reduce=None, reduce_concat=None
        )
        return genotype
    def genotypeno(self, i, task=''):
        def _parse(weights, layers):
            gene = []
            flag=0
            #print(len(weights))
            for i in range(layers):
                W = weights[i].copy()
                k_best = None
                #
                for k in range(len(W)):
                    #print('gene',k,PRIMITIVES[k])
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k
                if k_best==2 or k_best==3:
                    flag=flag+1
                if flag>1:
                    k_second_best = 0
                    for k in range(len(W)):
                        if W[k] > W[k_second_best] and k!=2 and k!=3:
                            k_second_best = k
                    k_best=k_second_best
                gene.append((PRIMITIVES[k_best], i))

            return gene

        if task == 'enhance':
            gene = _parse(F.softmax(self.alphas_enhances[i], dim=-1).data.cpu().numpy(), 2)
        elif task == 'hloss':
            gene = _parse(F.softmax(self.alphas_loss[i], dim=-1).data.cpu().numpy(), 2)
        elif task == 'denoise':#alphas_loss
            #print('denoise',self.alphas_denoises[i])
            gene = _parse(F.softmax(self.alphas_denoises[i], dim=-1).data.cpu().numpy(), 2)
        genotype = Genotype(
            normal=gene, normal_concat=None,
            reduce=None, reduce_concat=None
        )
        return genotype
    def genotypeorg(self, i, task=''):
        def _parse(weights, layers):
            gene = []
            for i in range(layers):
                W = weights[i].copy()
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k
                gene.append((PRIMITIVES[k_best], i))
            return gene

        if task == 'enhance':
            gene = _parse(F.softmax(self.alphas_enhances[i], dim=-1).data.cpu().numpy(), 7)
        elif task == 'hloss':
            gene = _parse(F.softmax(self.alphas_loss[i], dim=-1).data.cpu().numpy(), 7)
        elif task == 'denoise':#alphas_loss
            gene = _parse(F.softmax(self.alphas_denoises[i], dim=-1).data.cpu().numpy(), 7)
        genotype = Genotype(
            normal=gene, normal_concat=None,
            reduce=None, reduce_concat=None
        )
        return genotype
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
class LossFunctionl1(nn.Module):
    def __init__(self):
        super(LossFunctionl1, self).__init__()
        self.l2_loss = nn.L1Loss()#nn.MSELoss()
        # self.smooth_loss = SmoothLoss()

    def forward(self, output, target):
        #print(output.shape,target.shape) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 96, 96])
        return self.l2_loss(output, target)
class LossFunctionsmoothl1(nn.Module):
    def __init__(self):
        super(LossFunctionsmoothl1, self).__init__()
        self.l2_loss =nn.SmoothL1Loss()#nn.MSELoss()
    def forward(self, output, target):
        #print(output.shape,target.shape) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 96, 96])
        return self.l2_loss(output, target)
class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss,self).__init__()
    def forward(self,s,penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

class LossFunctiongradient(nn.Module):
    def __init__(self):
        super(LossFunctiongradient, self).__init__()

    def forward(self, x):

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
        #print(x.shape,grad_norm_x.shape,grad_norm_y.shape)


        # kernel_x = [[0, 0], [-1, 1]]
        # kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()
        #
        # kernel_y = [[0, -1], [0, 1]]
        # kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()
        #
        # weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        # weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        #
        # grad_x = torch.abs(F.conv2d(x, weight_x, padding=1))
        # grad_y = torch.abs(F.conv2d(x, weight_y, padding=1))
        #
        # grad_min_x = torch.min(grad_x)
        # grad_max_x = torch.max(grad_x)
        # grad_norm_x = torch.div((grad_x - grad_min_x), (grad_max_x - grad_min_x + 0.0001))
        #
        # grad_min_y = torch.min(grad_y)
        # grad_max_y = torch.max(grad_y)
        # grad_norm_y = torch.div((grad_y - grad_min_y), (grad_max_y - grad_min_y + 0.0001))
        return torch.mean(grad_norm_x)+torch.mean(grad_norm_y)
class LossFunctionl2(nn.Module):
    def __init__(self):
        super(LossFunctionl2, self).__init__()
        self.l2_loss = nn.MSELoss()
        # self.smooth_loss = SmoothLoss()

    def forward(self, output, target):
        #print(output.shape,target.shape) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 96, 96])
        return self.l2_loss(output, target)
class LossFunctionlP(nn.Module):
    def __init__(self):
        super(LossFunctionlP, self).__init__()
        # self.smooth_loss = SmoothLoss()

    def forward(self, output, target,p):
        no=torch.norm(target - output, p)
        no=torch.pow(no,p)
        no=no/(output.shape[0]*output.shape[1]*output.shape[2]*output.shape[3])
        #print(output.shape,target.shape) torch.Size([1, 3, 96, 96]) torch.Size([1, 3, 96, 96])
        return no

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()#做二元交叉熵并且求平均

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
            #print('real', target_tensor)#全1
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
            #print('false',target_tensor)#全0
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        #print('input',input)
        return self.loss(input, target_tensor)
#class LossClasser()
import cv2
def imsave(img, img_path):
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)
# convert torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())
class LossClasser(nn.Module):
    def __init__(self):
        super(LossClasser, self).__init__()
        '''self.vgg = VGG19_Extractor(output_layer_list=[2, 7, 16, 25])
        for v in self.vgg.parameters():
            v.requires_grad=False'''
        #self.l1=LossFunctionl1()
        self.Tensor = torch.cuda.FloatTensor #if self.gpu_ids else torch.Tensor
        #self.criterionGAN = GANLoss(use_lsgan=True,tensor=self.Tensor)
        # self.smooth_loss = SmoothLoss()
        #self.soft = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,output, target,discriminator):
        #mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis].astype('float32')
        std = np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis].astype('float32')
        # mean=np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
        mean = torch.from_numpy(mean).cuda()
        std = torch.from_numpy(std).cuda()
        x1 = tensor2uint(output)
        #print('???????',x1.size)
        #imsave(x1, '.\EXP/Cooperative-Searchclasserv5_2/p2p.png' )
        output = (output - mean) / std
        #input = input.cuda()
        #print(output.device)
        #print(output.shape)
        out = discriminator(output)

        #input_tensor = torch.tensor([[out[0][0], out[0][1]]])
        #print(out.shape,input_tensor.shape,input_tensor)  # ([1, 3])

        #m = self.soft(input_tensor)
        #print(m,torch.sum(m))
        #loss_D = m[0][0]
        label=torch.tensor([1]).cuda()
        #label=torch.tensor([[0,1],[0,1]]).cuda()
        #label2=torch.tensor([0]).cuda()
        #print(label.shape,out.shape) torch.Size([1]) torch.Size([1, 2])
        #print(out.shape,label.shape)
        #print(out,label)
        loss_D=self.criterion(out,label)
        #loss_D2=self.criterion(out,label2)
        #print('###########', out, loss_D,loss_D2)
        return loss_D,out
class LossDiscrim(nn.Module):
    def __init__(self):
        super(LossDiscrim, self).__init__()
        '''self.vgg = VGG19_Extractor(output_layer_list=[2, 7, 16, 25])
        for v in self.vgg.parameters():
            v.requires_grad=False'''
        #self.l1=LossFunctionl1()
        self.Tensor = torch.cuda.FloatTensor #if self.gpu_ids else torch.Tensor
        self.criterionGAN = GANLoss(use_lsgan=True,tensor=self.Tensor)
        # self.smooth_loss = SmoothLoss()

    def forward(self,output, target,discriminator):
        '''
        0-3 5-8 10-17 19-26
        4   4   8
        '''
        #print(output.shape,target.shape)
        #output_layer_list=[2,7,16,25]
        pred_fake =  discriminator(output)
        pred_real=  discriminator(target)
        #print(torch.mean(pred_fake),torch.mean(pred_real))
        loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +  # criterionGAN >0
                  self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
        #print(len(output_1_vgg)) 16
        #dloss= self.l1(output_1, target_1)
        #print('discriminatr',loss_D)0.6237 10.5098 #discriminatr tensor(nan, device='cuda:1', grad_fn=<DivBackward0>)
        #return 100*loss_D
        return loss_D


class LossFunctionvgg2(nn.Module):
    def __init__(self):
        super(LossFunctionvgg2, self).__init__()
        '''self.vgg = VGG19_Extractor(output_layer_list=[2, 7, 16, 25])
        for v in self.vgg.parameters():
            v.requires_grad=False'''
        self.l1=LossFunctionl1()
        # self.smooth_loss = SmoothLoss()

    def forward(self, output, target,vgg_model):
        output_1_vgg_1, output_1_vgg_2, output_1_vgg_3, output_1_vgg_4 =  vgg_model(output)
        output_2_vgg_1, output_2_vgg_2, output_2_vgg_3, output_2_vgg_4 =  vgg_model(target)
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

    def forward(self,vgg1, vgg2, vgg3, vgg4, output, target,vgg_model):
        '''
        0-3 5-8 10-17 19-26
        4   4   8
        '''
        #print(output.shape,target.shape)
        #output_layer_list=[2,7,16,25]
        output_1_vgg =  vgg_model(output)
        output_2_vgg =  vgg_model(target)
        #print(len(output_1_vgg)) 16
        vggloss=0
        for i in range(4):
            #print (i,vgg1[0][i])
            vggloss=vggloss+vgg1[0][i]*(self.l1(output_1_vgg[i], output_2_vgg[i]))
        for i in range(4):
            #print (i,vgg2[0][i])
            vggloss=vggloss+vgg2[0][i]*(self.l1(output_1_vgg[i+4], output_2_vgg[i+4]))
        for i in range(8):
            #print (i,vgg3[0][i])
            vggloss=vggloss+vgg3[0][i]*(self.l1(output_1_vgg[i+8], output_2_vgg[i+8]))
        for i in range(8):
            #print (i,vgg3[0][i])
            vggloss=vggloss+vgg4[0][i]*(self.l1(output_1_vgg[i+16], output_2_vgg[i+16]))
        '''for i in range(4):
            #print (i,vgg1[0][i])
            vggloss=vggloss+vgg1[0][i]*self.l1(output_1_vgg[i], output_2_vgg[i])
        for i in range(4):
            #print (i,vgg2[0][i])
            vggloss=vggloss+vgg2[0][i]*self.l1(output_1_vgg[i+4], output_2_vgg[i+4])
        for i in range(8):
            #print (i,vgg3[0][i])
            vggloss=vggloss+vgg3[0][i]*self.l1(output_1_vgg[i+8], output_2_vgg[i+8])
        for i in range(8):
            #print (i,vgg3[0][i])
            vggloss=vggloss+vgg4[0][i]*self.l1(output_1_vgg[i+16], output_2_vgg[i+16])'''

        #loss_vgg = loss_c_1 + loss_c_2 + loss_c_3 + loss_c_4
        return vggloss
class LossFunctionvggduibi(nn.Module):
    def __init__(self):
        super(LossFunctionvgg, self).__init__()
        '''self.vgg = VGG19_Extractor(output_layer_list=[2, 7, 16, 25])
        for v in self.vgg.parameters():
            v.requires_grad=False'''
        self.l1=LossFunctionl1()
        # self.smooth_loss = SmoothLoss()

    def forward(self,vgg1, vgg2, vgg3, vgg4, output, target,input,vgg_model):
        '''
        0-3 5-8 10-17 19-26
        4   4   8
        '''
        #print(output.shape,target.shape)
        #output_layer_list=[2,7,16,25]
        output_1_vgg =  vgg_model(output)
        output_2_vgg =  vgg_model(target)
        output_input_vgg = vgg_model(input)
        #print(len(output_1_vgg)) 16
        vggloss=0
        for i in range(4):
            #print (i,vgg1[0][i])
            vggloss=vggloss+vgg1[0][i]*(self.l1(output_1_vgg[i], output_2_vgg[i])/self.l1(output_input_vgg[i], output_1_vgg[i]))
        for i in range(4):
            #print (i,vgg2[0][i])
            vggloss=vggloss+vgg2[0][i]*(self.l1(output_1_vgg[i+4], output_2_vgg[i+4])/self.l1(output_input_vgg[i+4], output_1_vgg[i+4]))
        for i in range(8):
            #print (i,vgg3[0][i])
            vggloss=vggloss+vgg3[0][i]*(self.l1(output_1_vgg[i+8], output_2_vgg[i+8])/self.l1(output_input_vgg[i+8], output_1_vgg[i+8]))
        for i in range(8):
            #print (i,vgg3[0][i])
            vggloss=vggloss+vgg4[0][i]*(self.l1(output_1_vgg[i+16], output_2_vgg[i+16])/self.l1(output_input_vgg[i+16], output_1_vgg[i+16]))
        '''for i in range(4):
            #print (i,vgg1[0][i])
            vggloss=vggloss+vgg1[0][i]*self.l1(output_1_vgg[i], output_2_vgg[i])
        for i in range(4):
            #print (i,vgg2[0][i])
            vggloss=vggloss+vgg2[0][i]*self.l1(output_1_vgg[i+4], output_2_vgg[i+4])
        for i in range(8):
            #print (i,vgg3[0][i])
            vggloss=vggloss+vgg3[0][i]*self.l1(output_1_vgg[i+8], output_2_vgg[i+8])
        for i in range(8):
            #print (i,vgg3[0][i])
            vggloss=vggloss+vgg4[0][i]*self.l1(output_1_vgg[i+16], output_2_vgg[i+16])'''

        #loss_vgg = loss_c_1 + loss_c_2 + loss_c_3 + loss_c_4
        return vggloss
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
        self.input = input  # self.rgb2yCbCr(input)
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
