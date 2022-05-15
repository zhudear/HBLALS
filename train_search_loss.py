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
from model_search_loss import Network
#from model_search_classerv5 import Network
#from architectv4 import Architect
from multi_read_data import MemoryFriendlyLoader
from data.select_dataset import define_Dataset
from tools import utils_option as option
#from models.select_model import define_Model
from scipy import ndimage
from scipy.io import loadmat
#import utils_image as util
from Hyper_optimizer import HyperOptimizer
from torch.utils.tensorboard import SummaryWriter
#from DSFD.factory import build_net
#from discriminate_model import discriminatemodel
import visdom
import lpips
#python -m visdom.server
 # Loss plot
#viz = Visdom()
#viz = visdom.Visdom()
#viz = visdom.Visdom(port=8096)

#0.434478 0.039506 0.396413 5.163907 0.000000 0.400258 0.359333 0.491642
#epoch:15 step:0 total:9.321285 l1:0.087521 l2:0.034455 color:7.088637 ssim:1.864926 vgg:0.000000 smoothl1:0.056176 lp0.5:0.094817 lp0.8:0.094753 hyper:0.434478 0.039506 0.396413 5.163907 0.000000 0.400258 0.359333 0.491642
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
parser.add_argument('--arch_learning_rate', type=float, default=5e-4, help='learning rate for arch encoding')#  5e-4
parser.add_argument('--arch_weight_decay', type=float, default=1e-4, help='weight decay for arch encoding')#1e-3

parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='discriminator is here.')
parser.add_argument('--no_lsgan', action='store_true',help='do *not* use least square GAN, if false, use vanilla GAN')
parser.add_argument('--which_epoch', type=str, default='105',
                         help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--use_wgan', type=float, default=0, help='use wgan-gp')
parser.add_argument('--use_ragan', action='store_true',default=True, help='use ragan')
#parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

args = parser.parse_args()
if not os.path.isdir(r'./checkpoints/'):
    os.mkdir(r'./checkpoints/')
#checkpoints_path = r'./checkpoints/Cooperative-Searchclasserv5_2_3/'
checkpoints_path = r'./checkpoints/search_loss_LOL/'
checkpoints_path = r'./checkpoints/search_loss_underwater/'
checkpoints_path = r'./checkpoints/search_loss_dehaze/'
checkpoints_path = r'./checkpoints/search_loss_MIT_2/'
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
from data.dataset_usrnet import DatasetUSRNet_TXT
def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    writer = SummaryWriter("MIT_searchlog")
    json_path = 'options/train_usrnet2_2.json'
    #np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    #fth =r'./checkpoints\resnet18/underwater/epoch_1000.pth'
    #fth =r'./checkpoints\resnet18/dehaze/epoch_1000.pth'
    fth =r'./weights\resnet18/LLIE/epoch_560.pth'
    #checkp = torch.load(fth,map_location='cuda:1')
    checkp = torch.load(fth)
    #pretrain = torch.load('./weights/decom_net.pth',map_location='cpu') #加载到cpu
    #model.load_state_dict(pretrain)
    classer = checkp['model']  # 提取网络结构
    classer.load_state_dict(checkp['model_state_dict'])  # 加载网络权重参数
    classer=classer.cuda()
    for parameter in classer.parameters():
        parameter.requires_grad = False
    classer.eval()
    #classer=None




    #cudnn.benchmark = True
    #torch.manual_seed(args.seed)
    #cudnn.enabled = True
    #torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    logging.info("args = %s", args)
    logging.info("parser = %s", parser.parse_args())
    #ArgumentParser(prog='train_search.py', usage=None, description=None, formatter_class=<class 'argparse.HelpFormatter'>, conflict_handler='error', add_help=True)
    opt = option.parse(parser.parse_args().opt, is_train=True)
    #option.save(opt)
    opt = option.dict_to_nonedict(opt)
    logging.info("opt = %s", opt)
    # ----------------------------------------
    # seed
    # ----------------------------------------
    #seed = opt['train']['manual_seed']
    #if seed is None:
    seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    '''discriminator=discriminatemodel()
    discriminator.initialize(args)
    discriminator = discriminator.cuda()'''
    #model =  Network([1, 1, 0.1, 10, 1, 1,1,1])#,1])# Network()
    #lw=[1, 1, 0.1, 10, 1, 1,1,1,1,0.5]
    lw=[1, 1, 1, 1, 1, 1,1,1]
    logging.info("lw = %s", str(lw))

    model =  Network(lw,'/')#,1])# Network()
    '''pretrained_dict = torch.load(model_path+'/model_latest.pth')
    model.load_state_dict(pretrained_dict)'''

    '''with open('network.txt', 'a') as f:
        f.write(str(model))
    f.close()'''
    model = model.cuda()

    #使用torch.load加载训练好的参数
    '''pretrained_dict = torch.load( './checkpoints/model_257.pth')
    net2_dict = model.state_dict()
    dict=['hyper_1', 'hyper_2', 'hyper_3', 'hyper_4', 'hyper_5', 'hyper_6', 'hyper_7', 'hyper_8', 'hyper_9']
    #print('net2_dict',net2_dict.keys())
    #只保留net2中需要的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net2_dict and k not in dict}
    #print('pretrained_dict',pretrained_dict.keys())
    #使用update更新参数
    net2_dict.update(pretrained_dict)
    #使用load_state_dict将参数加载到net2中
    model.load_state_dict(net2_dict)'''

    #model.load_state_dict(torch.load(model_path + 'model_257.pth'))
    print('loaded!')
    vgg_model = VGG19_Extractor()#output_layer_list=[2,7,16,25]
    #vgg_model = VGG19_Extractor2(output_layer_list=[2,7,16,25])  # [2,5,13,25]
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
    '''logging.info("requires_grad:")
    #all of the parameters
    for k,v in model.named_parameters():
        if v.requires_grad:
            logging.info("%s", k)'''


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
        if phase == 'trainMIT':#trainunderwater trainLOL  traindehaze
            TrainDataset = DatasetUSRNet_TXT(dataset_opt,True)
            train_queue = torch.utils.data.DataLoader(
                TrainDataset, batch_size=1,#args.batch_size,
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=1,  # dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            print('train',len(train_queue))#50
        elif phase == 'searchMIT':
            ValidDataset = DatasetUSRNet_TXT(
                dataset_opt,True)  # MemoryFriendlyLoader(img_dir=valid_low_data_names, task='valid')

            valid_queue = torch.utils.data.DataLoader(
                ValidDataset, batch_size=1,#args.batch_size,
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=1,  # dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            print('vail', len(valid_queue))#50
        elif phase == 'testMIT':
            Testset = DatasetUSRNet_TXT(
                dataset_opt, False)  # MemoryFriendlyLoader(img_dir=valid_low_data_names, task='valid')
            test_loader = torch.utils.data.DataLoader(
                Testset, batch_size=1,  # args.batch_size,
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=1,  # dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            print('test', len(test_loader))  # 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(50), eta_min=args.learning_rate_min)
    #architect = Architect(model, args)
    #hyper_optimizer=[]
    hyper_optimizer = HyperOptimizer(model, args)
    #start=0
    '''viz.line([model.hyper_1.item()], [start], win='l1-loss-', name='l1', update='append',
             opts=dict(title='l1'))
    viz.line([model.hyper_2.item()], [start], win='l2-loss-', name='l2', update='append',
             opts=dict(title='l2'))
    viz.line([model.hyper_3.item()], [start], win='color-loss-', name='color', update='append',
             opts=dict(title='color'))
    viz.line([model.hyper_4.item()], [start], win='ssim-loss-', name='ssim', update='append',
             opts=dict(title='ssim'))
    viz.line([model.hyper_5.item()], [start], win='vgg-loss-', name='vgg', update='append',
             opts=dict(title='vgg'))'''
    '''viz.line([model.hyper_6.item()], [start], win='smoothl1-loss-', name='smoothl1', update='append',
             opts=dict(title='smoothl1'))
    viz.line([model.hyper_7.item()], [start], win='lp0.5-loss-', name='lp0.5', update='append',
             opts=dict(title='lp0.5'))
    viz.line([model.hyper_8.item()], [start], win='lp0.8-loss-', name='lp0.8', update='append',
             opts=dict(title='lp0.8'))
    viz.line([model.hyper_9.item()], [start], win='gradient-loss-', name='gradient', update='append',
             opts=dict(title='gradient'))'''

    for epoch1 in range(args.epochs):
        epoch=epoch1
        lr = scheduler.get_lr()
        logging.info('epoch %d lr %e', epoch, lr[0])
        vgg1=F.softmax(model.vgg1, dim=-1)
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
        logging.info('vgg4:%d %s', ReturnIndices.item() + 19, F.softmax(vgg4, dim=-1))
        #discriminator
        train(train_queue, valid_queue,test_loader, model,hyper_optimizer,vgg_model,loss_fn_alex,classer , optimizer, lr,writer, epoch)#architect
        #train(train_queue, valid_queue,test_loader, model,architect,hyper_optimizer, optimizer, lr, epoch,isupdate)
        model_save_path = os.path.join(model_path, 'model_latest.pth')
        torch.save(model.state_dict(), model_save_path)
        '''if epoch > 30 and epoch<100 and epoch % 10 == 0:
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
    if img.ndim == 3:#RGB 012  BGR 210
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)
# convert torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

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

    x1 = tensor2uint(x)#output,reflection,Illimination1
    imsave(x1, path + '\%s_%s_%d_out.png' % (epoch, step,coun))
    '''x1 = tensor2uint(x[1])  # output,reflection,Illimination1
    imsave(x1, path + '\%s_%s_%d_reflection.png' % (epoch, step, coun))
    x1 = tensor2uint(x[2])  # output,reflection,Illimination1
    imsave(x1, path + '\%s_%s_%d_Illimination.png' % (epoch, step, coun))'''

def train(train_queue, valid_queue,test_loader, model,hyper_optimizer,vgg_model,loss_fn_alex, discriminator,optimizer, lr,writer, epoch):#architect
    '''l1=0.0
    l2=0.0
    color=0.0
    ssim=0.0
    vgg=0.0
    total=0.0
    lp=0.0'''
    classerloss=0
    tlos=0
    for step, (input1) in enumerate(train_queue):
        #step=step+1
        #print(step)
        #return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
        model.train()
        target =input1['H'].cuda()# input.clone()
        input = input1['L'].cuda()#F.interpolate(input, scale_factor=0.25)
        # input_search.clone()

        #first update the α of hloss network
     #update the w of hloss net
        ''' model.hyper_1.requires_grad = False
        model.hyper_2.requires_grad = False
        model.hyper_3.requires_grad = False
        model.hyper_4.requires_grad = False'''
        _search = next(iter(valid_queue))
        input_search = _search['L'].cuda()  # torch.tensor(input_search, requires_grad=False).cuda()
        target_search = _search['H'].cuda()
        #print('%%%%%%%%%%%%%',target.shape,input.shape,input_search.shape,target_search.shape)
        #print('train', input1['L_path'],'search',_search['L_path'])
        # _search2 = next(iter(valid_queue2))
        # input_search2 = _search2['L'].cuda()  # torch.tensor(input_search, requires_grad=False).cuda()
        # target_search2 = _search2['H'].cuda()
        '''logging.info('before %f %f %f %f',  model.hyper_1.item(), model.hyper_2.item(),
                     model.hyper_3.item(), model.hyper_4.item())'''
        #print('hyper_optimizer.step')
        #if True:#isupdate == 2:
        '''logging.info('train epoch:%d step:%d '
                     'hyper:%f %f %f %f %f '
                     '%f %f %f '
                     ,
                     epoch, step,
                     model.hyper_1.item(), model.hyper_2.item(), model.hyper_3.item(), model.hyper_4.item(),
                     model.hyper_5.item()
                     , model.hyper_6.item(), model.hyper_7.item(), model.hyper_8.item())'''
        model.hyper_1.requires_grad = True
        model.hyper_2.requires_grad = True
        model.hyper_3.requires_grad = True
        model.hyper_4.requires_grad = True
        model.hyper_5.requires_grad = True
        model.hyper_6.requires_grad = True
        model.hyper_7.requires_grad = True
        model.hyper_8.requires_grad = True
        model.vgg1.requires_grad = True
        model.vgg2.requires_grad = True
        model.vgg3.requires_grad = True
        model.vgg4.requires_grad = True
        #model.hyper_9.requires_grad = True
        # #print('EEE')
        hyper_optimizer.step(input, target,input_search, target_search, lr, vgg_model,loss_fn_alex,discriminator, optimizer, unrolled=True)
        if model.hyper_1<0:
            model.hyper_1.data.fill_(0.0)
        if model.hyper_2<0:
            model.hyper_2.data.fill_(0.0)
        if model.hyper_3<0:
            model.hyper_3.data.fill_(0.0)
        if model.hyper_4<0:
            model.hyper_4.data.fill_(0.0)
        if model.hyper_5<0:
            model.hyper_5.data.fill_(0.0)

        if model.hyper_6<0:
            model.hyper_6.data.fill_(0.0)
        if model.hyper_7<0:
            model.hyper_7.data.fill_(0.0)
        if model.hyper_8<0:
            model.hyper_8.data.fill_(0.0)


        model.hyper_1.requires_grad = False
        model.hyper_2.requires_grad = False
        model.hyper_3.requires_grad = False
        model.hyper_4.requires_grad = False
        model.hyper_5.requires_grad = False
        model.hyper_6.requires_grad = False
        model.hyper_7.requires_grad = False
        model.hyper_8.requires_grad = False


        model.vgg1.requires_grad = False
        model.vgg2.requires_grad = False
        model.vgg3.requires_grad = False
        model.vgg4.requires_grad = False
        #model.hyper_9.requires_grad = False
        #architect.step(input, target, input_search, target_search, lr, vgg_model, discriminator,optimizer, unrolled=True)
        optimizer.zero_grad()
        totalloss = model._loss(input, target,vgg_model,loss_fn_alex,discriminator,1)
        #1:use w*lossi
        #2:use gan
        loss=totalloss[0]
        loss.backward()
        tlos+=loss
        #print('11111111111')
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        clas = model._loss(input, target, vgg_model,loss_fn_alex, discriminator, 2)
        #print(clas)
        classerloss+=clas[0]
        if step % args.report_freq == 0:

            #print(clas)
            logging.info('%s',str(clas))
            #logging.info('train epoch:%d step:%d total:%f ',epoch, step, loss)
            #print(totalloss[1])
            #logging.info('train epoch:%d step:%d total:%f ',epoch, step,loss)

            print(input1['L_path'],input1['H_path'], _search['L_path'],_search['H_path'],input.shape)
            logging.info(
                'train epoch:%d step:%d total:%f l1:%f l2:%f color:%f ssim:%f vgg:%f lpips:%f  tv:%f lap:%f '

                'hyper:%f %f %f %f %f %f %f '
                '%f'
                ,
                epoch, step, loss, totalloss[1], totalloss[2], totalloss[3], totalloss[4], totalloss[5],
                totalloss[6], totalloss[7], totalloss[8],
                model.hyper_1.item(), model.hyper_2.item(), model.hyper_3.item(), model.hyper_4.item(),
                model.hyper_5.item()
                , model.hyper_6.item(), model.hyper_7.item(), model.hyper_8.item()
                )
        if  step%len(train_queue)==0 :
            print('step',epoch,step,step%len(train_queue),epoch+float(step)/len(train_queue))
            # print(clas)
            #print('target',target)
            #print('input',input)
            logging.info('%s#############%s', str(tlos),str(classerloss))
            '''viz.line([classerloss.item()], [epoch + float(step) / len(train_queue)], win='classer-no-loss', name='classerMITno',
                     update='append',
                     opts=dict(title='classer'))
            viz.line([ tlos.item()], [epoch + float(step) / len(train_queue)], win='train-MIT-no-loss', name='trainMITno',
                     update='append',
                     opts=dict(title='train'))'''
            classerloss=0
            tlos=0
            '''viz.line([model.hyper_1.item()], [epoch+float(step)/len(train_queue)], win='l1-loss', name='l1', update='append', opts=dict(title='l1'))
            viz.line([model.hyper_2.item()], [epoch+float(step)/len(train_queue)], win='l2-loss', name='l2', update='append', opts=dict(title='l2'))
            viz.line([model.hyper_3.item()], [epoch+float(step)/len(train_queue)], win='color-loss', name='color', update='append', opts=dict(title='color'))
            viz.line([model.hyper_4.item()], [epoch+float(step)/len(train_queue)], win='ssim-loss', name='ssim', update='append', opts=dict(title='ssim'))
            viz.line([model.hyper_5.item()], [epoch+float(step)/len(train_queue)], win='vgg-loss', name='vgg', update='append', opts=dict(title='vgg'))
            viz.line([model.hyper_6.item()], [epoch+float(step)/len(train_queue)], win='lpips-loss', name='lpips', update='append', opts=dict(title='lpips'))
            viz.line([model.hyper_7.item()], [epoch+float(step)/len(train_queue)], win='gw-loss', name='gw', update='append', opts=dict(title='gw'))
            viz.line([model.hyper_8.item()], [epoch+float(step)/len(train_queue)], win='lp0.8-loss', name='lp0.8', update='append', opts=dict(title='lp0.8'))'''
            #viz.line([model.hyper_9.item()], [epoch+float(step)/len(train_queue)], win='gradient-loss-', name='gradient', update='append', opts=dict(title='gradient'))

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
    writer.add_scalar('add_scalar/totalloss', tlos, epoch)
    writer.add_scalars('add_scalars/loss', {'l1':totalloss[1],'l2':totalloss[2],'color':totalloss[3],
                                           'ssim':totalloss[4],'vgg':totalloss[5],'lpips':totalloss[6],'tv':totalloss[7],'lap':totalloss[8]}, epoch)
    writer.add_scalars('add_scalars/hyper', {'l1': model.hyper_1.item(), 'l2': model.hyper_2.item(), 'color': model.hyper_3.item(),
                                           'ssim': model.hyper_4.item(), 'vgg': model.hyper_5.item(), 'lpips': model.hyper_6.item(),
                                           'tv': model.hyper_7.item(), 'lap': model.hyper_8.item(),}, epoch)
if __name__ == '__main__':
    main()
