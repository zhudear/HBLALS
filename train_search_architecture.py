import os
import sys
import numpy as np
import torch
from tools import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from skimage.io import imsave
import cv2
#import utils_sisr as sisr
import matplotlib.pyplot as plt
from losses.vgg import *
from PIL import Image
from torch.autograd import Variable
from model_search_architecture  import Network#model_search_classerv4   model_search_classerv4_3_2
from architect import Architect
from multi_read_data import MemoryFriendlyLoader
#from data.select_dataset import define_Dataset
from tools import utils_option as option
#from models.select_model import define_Model
from scipy import ndimage
from scipy.io import loadmat
from tools import utils_image as util
from data.dataset_usrnet import DatasetUSRNet_TXT
#from Hyper_optimizer import HyperOptimizer
#from DSFD.factory import build_net
#from discriminate_model import discriminatemodel
import visdom
import lpips
#python -m visdom.server
 # Loss plot
#viz = Visdom()
#viz = visdom.Visdom()
# = visdom.Visdom(port=8096)

parser = argparse.ArgumentParser("ruas")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='init learning rate')#0.005
parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='checkpoints', help='checkpointseriment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=5e-4, help='learning rate for arch encoding')#3e-3
parser.add_argument('--arch_weight_decay', type=float, default=1e-4, help='weight decay for arch encoding')#1e-3

parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='discriminator is here.')
parser.add_argument('--no_lsgan', action='store_true',help='do *not* use least square GAN, if false, use vanilla GAN')
parser.add_argument('--which_epoch', type=str, default='105',
                         help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--use_wgan', type=float, default=0, help='use wgan-gp')
parser.add_argument('--use_ragan', action='store_true',default=True, help='use ragan')
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

args = parser.parse_args()
if not os.path.isdir(r'./checkpoints/'):
    os.mkdir(r'./checkpoints/')
checkpoints_path = r'./checkpoints/search_A_LOL/'
checkpoints_path = r'./checkpoints/search_A_underwater/'
checkpoints_path = r'./checkpoints/search_A_MIT/'
checkpoints_path = r'./checkpoints/search_A_LOL_vgg/'
if not os.path.isdir(checkpoints_path):
    os.mkdir(checkpoints_path)
inference_dir = checkpoints_path + '/inference/'
if not os.path.isdir(inference_dir):
    os.mkdir(inference_dir)
model_path = checkpoints_path + '\model/'
if not os.path.isdir(model_path):
    os.mkdir(model_path)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(checkpoints_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
class TestDataset():
    '''
    # -----------------------------------------
    # Get L/k/sf/sigma for USRNet.
    # Only "paths_H" and kernel is needed, synthesize L on-the-fly.
    # -----------------------------------------
    '''
    def __init__(self):
        super(TestDataset, self).__init__()
        self.n_channels =  3

        self.sigma_max = 25#self.opt['sigma_max'] if self.opt['sigma_max'] is not None else 25

        self.sf_validation = 4
        #self.kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']
        #self.kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']  # for validation

        # -------------------
        # get the path of H
        # -------------------
        self.paths_H = util.get_image_paths(r'./trainsets\test')  # return None if input is None
        self.count = 0

    def __getitem__(self, index):

        # -------------------
        # get H image
        # -------------------
        H_path = self.paths_H[index]
        #print('H_path',H_path)#H_path trainsets/trainH\0016.png
        img_H = util.imread_uint(H_path, self.n_channels)
        L_path = H_path
        #self.opt['dataloader_batch_size']=1
        '''if self.count % self.opt['dataloader_batch_size'] == 0:
            # sf = random.choice([1,2,3,4])
            self.sf = random.choice(self.scales)'''
        self.sf =4# random.choice(self.scales)
            # self.count = 0  # optional
        #print('self.sf1', self.sf)
        self.count += 1
        #self.sf = self.sf_validation

        #k = self.kernels[0, 0].astype(np.float64)  # validation kernel
        #k /= np.sum(k)
        '''noise_level = 0. / 255.0  # validation noise level
        img_L = ndimage.filters.convolve(img_H, np.checkpointsand_dims(k, axis=2), mode='wrap')  # blur
        img_L = img_L[0::self.sf_validation, 0::self.sf_validation, ...]  # downsampling
        img_L = util.uint2single(img_L) + np.random.normal(0, noise_level, img_L.shape)'''

        #k = util.single2tensor3(np.checkpointsand_dims(np.float32(k), axis=2))
        img_H, img_L = util.uint2tensor3(img_H), util.single2tensor3(img_L)
        #noise_level = torch.FloatTensor([noise_level]).view([1,1,1])
        return {'L': img_L, 'H': img_H,  'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    json_path = 'options/train_usrnet2_2.json'
    #np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)



    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    logging.info("parser = %s", parser)
    #ArgumentParser(prog='train_search.py', usage=None, description=None, formatter_class=<class 'argparse.HelpFormatter'>, conflict_handler='error', add_help=True)
    opt = option.parse(parser.parse_args().opt, is_train=True)
    #option.save(opt)
    opt = option.dict_to_nonedict(opt)


    seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #lol 1.376733 1.280910 0.821452 1.225962 0.490507 0.743446 0.000000 0.000000
    lw=[1.38,1.28,0.82,1.23,0.49,0.74,0,0]
    #
    #underwater 1.113956 1.091780 1.395290 1.895177 1.209277 1.093341 0.000324 0.000381
    #lw=[1.11, 1.09,1.40, 1.90, 1.21, 1.09,0,0]
    #MIT 1.274806 1.129426 0.852978 1.485702 1.156918 0.962715 0.000251 0.153165
    #lw=[1.27, 1.13, 0.85, 1.49, 1.16, 0.96, 0, 0.15]

    logging.info("learned lw = %s", lw)
    model =  Network(lw,'/') #epoch 15
    model = model.cuda()
    #model.load_state_dict(torch.load(model_path + 'model_latest.pth'))
    '''pretrained_dict =torch.load( './checkpoints/model_257.pth')# torch.load(model_path + 'model_257.pth')
    net2_dict = model.state_dict()
    dict = ['hyper_1', 'hyper_2', 'hyper_3', 'hyper_4', 'hyper_5', 'hyper_6', 'hyper_7', 'hyper_8', 'hyper_9']
    # print('net2_dict',net2_dict.keys())
    # ?????????net2??????????????????
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net2_dict and k not in dict}
    # print('pretrained_dict',pretrained_dict.keys())
    # ??????update????????????
    net2_dict.update(pretrained_dict)
    # ??????load_state_dict??????????????????net2???
    model.load_state_dict(net2_dict)'''

    print('loaded!')
    #vgg_model = VGG19_Extractor()#output_layer_list=[2,7,16,25]
    #vgg_model =VGG19_Extractor2(output_layer_list=[3, 8, 15, 21]) underwater  # [2,5,13,25]
    vgg_model =VGG19_Extractor2(output_layer_list=[1, 6, 11,22])  # [2,5,13,25]
    vgg_model=vgg_model.cuda()
    for parameter in vgg_model.parameters():
        parameter.requires_grad = False
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.cuda()
    for k in loss_fn_alex.parameters():
        k.requires_grad = False
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    '''for k,v in model.named_parameters():
        if v.requires_grad:
            print(k)'''

    #all of the parameters

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum)

    # prepare DataLoader
    #train_low_data_names = r'F:\paper_reading\2021\trainB/*.png'
    # train_low_data_names = r'H:\image-enhance\UPE500\trainA/*.png'
    #TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')
    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        #dataset_opt['dataloader_batch_size'] = 1
        #dataset_opt['dataloader_num_workers'] = 0
        #print(phase,'dataset_opt',dataset_opt)

        # -----------------------------------------
        # common
        # -----------------------------------------
        #train dataset_opt {'name': 'train_dataset', 'dataset_type': 'usrnet', 'dataroot_H': 'trainsets/trainH', 'dataroot_L': None, 'H_size': 96, 'use_flip': True,
        # 'use_rot': True, 'scales': [1, 2, 3, 4], 'dataloader_shuffle': True, 'dataloader_num_workers': 8, 'dataloader_batch_size': 48, 'phase': 'train', 'scale': 4, 'n_channels': 3}
        if phase == 'trainLOL':#trainunderwater
            TrainDataset = DatasetUSRNet_TXT(dataset_opt,True)
            train_queue = torch.utils.data.DataLoader(
                TrainDataset, batch_size=1,#args.batch_size,
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=1,  # dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            print('train',len(train_queue))#50
        elif phase == 'searchLOL':
            ValidDataset = DatasetUSRNet_TXT(
                dataset_opt,True)  # MemoryFriendlyLoader(img_dir=valid_low_data_names, task='valid')

            valid_queue = torch.utils.data.DataLoader(
                ValidDataset, batch_size=1,#args.batch_size,
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=1,  # dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            print('vail', len(valid_queue))#50
        elif phase == 'testLOL':
            Testset = DatasetUSRNet_TXT(
                dataset_opt, False)  # MemoryFriendlyLoader(img_dir=valid_low_data_names, task='valid')
            test_loader = torch.utils.data.DataLoader(
                Testset, batch_size=1,  # args.batch_size,
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=1,  # dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            print('test', len(test_loader))  # 50

        #valid_low_data_names = r'F:\paper_reading\2021\validB/*.png'
        # valid_low_data_names = r'H:\image-enhance\UPE500\validA/*.png'
        # ValidDataset = MemoryFriendlyLoader(img_dir=valid_low_data_names, task='valid')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(50), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    #hyper_optimizer = HyperOptimizer(model, args)
    for epoch1 in range(args.epochs):
        epoch=epoch1#+5
        lr = scheduler.get_lr()
        logging.info('epoch %d lr %e', epoch, lr[0])
        logging.info('Architect of NRM:')
        logging.info('nrm = %s', str(0))
        genotype = model.genotype(0, task='denoise')
        logging.info('genotype0 = %s', genotype)
        genotype = model.genotype(1, task='denoise')
        logging.info('genotype1 = %s', genotype)
        genotype = model.genotype(2, task='denoise')
        logging.info('genotype2 = %s', genotype)
        genotype = model.genotype(3, task='denoise')
        logging.info('genotype3 = %s', genotype)
        genotype = model.genotype(4, task='denoise')
        logging.info('genotype4 = %s', genotype)
        genotype = model.genotype(5, task='denoise')
        logging.info('genotype5 = %s', genotype)
        genotype = model.genotype(6, task='denoise')
        logging.info('genotype6 = %s', genotype)

        genotype = model.genotype(0, task='e')
        logging.info('genotypee = %s', genotype)
        genotype = model.genotype(0, task='d')
        logging.info('genotyped = %s', genotype)

        logging.info('nrm %s', str(0))
        logging.info('0:%s', F.softmax(model.alphas_denoises[0], dim=-1))
        logging.info('1:%s', F.softmax(model.alphas_denoises[1], dim=-1))
        logging.info('2:%s', F.softmax(model.alphas_denoises[2], dim=-1))
        logging.info('3:%s', F.softmax(model.alphas_denoises[3], dim=-1))
        logging.info('4:%s', F.softmax(model.alphas_denoises[4], dim=-1))
        logging.info('5:%s', F.softmax(model.alphas_denoises[5], dim=-1))
        logging.info('6:%s', F.softmax(model.alphas_denoises[6], dim=-1))

        logging.info('e:%s', F.softmax(model.alphas_e[0], dim=-1))
        logging.info('d:%s', F.softmax(model.alphas_d[0], dim=-1))
        '''vgg1=F.softmax(model.vgg1, dim=-1)
        #print(vgg1.shape,vgg1)
        ReturnVlaue, ReturnIndices = vgg1.max(1)
        #print(ReturnVlaue, ReturnIndices.item() )
        #index=vgg1.index(max(vgg1))
        logging.info('vgg1:%d %s', ReturnIndices.item(),vgg1 )
        vgg2 = F.softmax(model.vgg2, dim=-1)
        ReturnVlaue, ReturnIndices = vgg2.max(1)
        #index = vgg2.index(max(vgg2))
        logging.info('vgg2:%d %s', ReturnIndices.item()+5, vgg2)

        vgg3 = F.softmax(model.vgg3, dim=-1)
        ReturnVlaue, ReturnIndices = vgg3.max(1)
        #index = vgg3.index(max(vgg3))
        logging.info('vgg3:%d %s', ReturnIndices.item()+10, F.softmax(vgg3, dim=-1))

        vgg4 = F.softmax(model.vgg4, dim=-1)
        ReturnVlaue, ReturnIndices = vgg4.max(1)
        # index = vgg3.index(max(vgg3))
        logging.info('vgg4:%d %s', ReturnIndices.item() + 19, F.softmax(vgg4, dim=-1))'''
        #discriminator
        train(train_queue, valid_queue,test_loader, model,architect,vgg_model ,loss_fn_alex,None, optimizer, lr, epoch)#architect
        model_save_path = os.path.join(model_path, 'model_latest.pth')
        torch.save(model.state_dict(), model_save_path)
        #train(train_queue, valid_queue,test_loader, model,architect,hyper_optimizer, optimizer, lr, epoch,isupdate)
        '''model_save_path = os.path.join(model_path, 'model_latest.pth')
        torch.save(model.state_dict(), model_save_path)
        if epoch > 30 and epoch<100 and epoch % 10 == 0:
            model_save_path = os.path.join(model_path, 'model_%d.pth' % epoch)
            torch.save(model.state_dict(), model_save_path)
        elif epoch>100 and epoch%5==0:
            model_save_path = os.path.join(model_path, 'model_%d.pth' % epoch)
            torch.save(model.state_dict(), model_save_path)'''

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]
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
def infer4(net_input,target, model, epoch,step,coun):
  with torch.no_grad():
    #print('infer2', epoch, step)
    x= model(net_input)
    #print('out',len(x))#16
    path = r'.\checkpoints\Cooperative-Search\inference'
    '''input = tensor2uint(net_input)
    imsave(input, path + '\%s_%s_%dx_input.png' % (epoch, step,coun))
    target = tensor2uint(target)
    imsave(target, path + '\%s_%s_%d_gt.png' % (epoch,step ,coun))'''
    x1 = tensor2uint(x[-1])
    imsave(x1, path + '\%s_%s_%d.png' % (epoch, step,coun))
def infer(net_input,target, model, epoch,step,coun):
  with torch.no_grad():
    #print('infer2', epoch, step)
    x= model(net_input)
    print('out',len(x))#3
    path = r'.\checkpoints\Cooperative-Search\inference'
    '''input = tensor2uint(net_input)
    imsave(input, path + '\%s_%s_%dx_input.png' % (epoch, step,coun))
    target = tensor2uint(target)
    imsave(target, path + '\%s_%s_%d_gt.png' % (epoch,step ,coun))'''
    for i in range(len(x)):
        x1 = tensor2uint(x[i])#output,reflection,Illimination1
        imsave(x1, path + '\%s_%s_%d_%d.png' % (epoch, step,coun,i))


def lowinfer(net_input,target, model, epoch,step,coun):
  with torch.no_grad():
    #print('infer2', epoch, step)
    x= model(net_input)
    #print('out',len(x))#3
    path =inference_dir
    '''input = tensor2uint(net_input)
    imsave(input, path + '\%s_%s_%dx_input.png' % (epoch, step,coun))
    target = tensor2uint(target)
    imsave(target, path + '\%s_%s_%d_gt.png' % (epoch,step ,coun))'''

    #x3 = tensor2uint(net_input)#output,reflection,Illimination1
    x1 = tensor2uint(x)#output,reflection,Illimination1
    #x2 = tensor2uint(target)#output,reflection,Illimination1
    imsave(x1, path + '\%s_%s_%d_out.png' % (epoch, step,coun))
    #imsave(x2, path + '\%s_%s_%d_target.png' % (epoch, step,coun))
    #imsave(x3, path + '\%s_%s_%d_input.png' % (epoch, step,coun))
    '''x1 = tensor2uint(x[1])  # output,reflection,Illimination1
    imsave(x1, path + '\%s_%s_%d_reflection.png' % (epoch, step, coun))
    x1 = tensor2uint(x[2])  # output,reflection,Illimination1
    imsave(x1, path + '\%s_%s_%d_Illimination.png' % (epoch, step, coun))'''
def infer3(net_input,target, model, epoch,step,coun):
  with torch.no_grad():
    #print('infer2', epoch, step)
    x= model(net_input)
    #print('out',len(x))#16
    path = r'.\checkpoints\Cooperative-Search\inference'
    '''input = tensor2uint(net_input)
    imsave(input, path + '\%d_input.png' % (epoch,coun))
    target = tensor2uint(target)
    imsave(target, path + '\%d_gt.png' % (epoch,coun))'''
    for i in range(len(x)):
        x1 = tensor2uint(x[i])
        iteration=i/4+1
        if(i%4==0):#d
            imsave(x1, path + '\%s_%d_%d_d.png' % (epoch,coun, iteration))
        elif (i % 4 == 1):  # d
            imsave(x1, path + '\%s_%d_%d_d_res.png' % (epoch,coun, iteration))
        elif (i % 4 == 2):  # d
            imsave(x1, path + '\%s_%d_%d_p.png' % (epoch,coun, iteration))
        else:
            imsave(x1, path + '\%s_%d_%d_p_res.png' % (epoch,coun, iteration))
def infer2(net_input,target, model, epoch,step,coun):
  with torch.no_grad():
    #print('infer2', epoch, step)
    x= model(net_input)
    #print('out',len(x))#16
    path =inference_dir
    input = tensor2uint(net_input)
    imsave(input, path + '\%s_%s_%dx_input.png' % (epoch, step,coun))
    target = tensor2uint(target)
    imsave(target, path + '\%s_%s_%d_gt.png' % (epoch,step ,coun))
    for i in range(len(x)):
        x1 = tensor2uint(x[i])
        iteration=i/3+1
        if(i%3==0):#d
            imsave(x1, path + '\%s_%d_d.png' % (epoch, iteration))
        elif (i % 3 == 1):  # d
            imsave(x1, path + '\%s_%d_p.png' % (epoch, iteration))
        else:
            imsave(x1, path + '\%s_%d_res.png' % (epoch, iteration))
def zero(x):
    return (abs(x) + x) / 2
def train(train_queue, valid_queue,test_loader, model,architect,vgg_model,loss_fn_alex, discriminator,optimizer, lr, epoch):#architect
    '''l1=0.0
    l2=0.0
    color=0.0
    ssim=0.0
    vgg=0.0
    total=0.0
    lp=0.0'''
    tlos = 0
    valid_=iter(valid_queue)
    for step, (input1) in enumerate(train_queue):
        #print(step)
        #return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
        model.train()
        target =input1['H'].cuda()# input.clone()
        input = input1['L'].cuda()#F.interpolate(input, scale_factor=0.25)
        # input_search.clone()
        '''ii=tensor2uint(input)
        imsave(ii, './checkpoints\in.png' )
        ii = tensor2uint(target)
        imsave(ii, './checkpoints\itarget.png')'''

        #first update the ?? of hloss network
     #update the w of hloss net
        ''' model.hyper_1.requires_grad = False
        model.hyper_2.requires_grad = False
        model.hyper_3.requires_grad = False
        model.hyper_4.requires_grad = False'''
        _search = next(valid_)
        input_search = _search['L'].cuda()  # torch.tensor(input_search, requires_grad=False).cuda()
        target_search = _search['H'].cuda()

        architect.step(input, target, input_search, target_search, lr, vgg_model, loss_fn_alex,discriminator,optimizer, unrolled=True)
        optimizer.zero_grad()
        totalloss = model._loss(input, target,vgg_model,loss_fn_alex,discriminator,1)
        #1:use w*lossi
        #2:use gan
        loss=totalloss[0]
        loss.backward()
        tlos += loss
        #print('11111111111')
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        '''total+=totalloss[0]
        l1+=totalloss[1]
        l2+=totalloss[2]
        color+=totalloss[3]
        ssim+=totalloss[4]
        vgg+=totalloss[5]
        lp+=totalloss[6]'''
        #print('OOOO')

        '''logging.info('after weight: %f %f %f %f', model.hyper_1.item(), model.hyper_2.item(),
                     model.hyper_3.item(), model.hyper_4.item())'''

        if step % args.report_freq == 0:

            '''logging.info('weights: %f %f %f %f', model.hyper_1.item(), model.hyper_2.item(),
                         model.hyper_3.item(), model.hyper_4.item())'''
            logging.info(
                'train epoch:%d step:%d total:%f l1:%f l2:%f color:%f ssim:%f vgg:%f lpips:%f tv:%f lap:%f'

                'hyper:%f %f %f %f %f %f %f '
                '%f '
                ,
                epoch, step, loss, totalloss[1], totalloss[2], totalloss[3], totalloss[4], totalloss[5],
                totalloss[6], totalloss[7], totalloss[8],
                model.hyper_1.item(), model.hyper_2.item(), model.hyper_3.item(), model.hyper_4.item(),
                model.hyper_5.item()
                , model.hyper_6.item(), model.hyper_7.item(), model.hyper_8.item()
                )

        if  step%len(train_queue)==0 :
            '''viz.line([tlos.item()], [epoch + float(step) / len(train_queue)], win='train-loss2', name='train',
                     update='append',
                     opts=dict(title='train'))'''
            tlos = 0
            coun=0
            for test_data in test_loader:
                '''for item in test_data:
                    if (item == 'sigma'):
                        testsigma = test_data[item].cuda()'''
                    # print(item,type(input[item]))
                    # list1.append(input[item])
                # print('feed_data', input['H'].shape, inputsigma.shape)
                # feed_data torch.Size([1, 3, 96, 96]) torch.Size([1, 1, 1, 1])

                # input = torch.tensor(input, requires_grad=False).cuda()
                ttarget = test_data['H'].cuda()  # input.clone()
                test = test_data['L'].cuda()
                #print('test', ttarget.shape, test.shape)
                lowinfer(test,ttarget,model,epoch,step,coun)#4
                coun=coun+1

if __name__ == '__main__':
    main()
