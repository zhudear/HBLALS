import os
import sys
import numpy as np
import torch
from tools import utils
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import random
from losses.vgg import VGG19_Extractor2
from PIL import Image
from torch.autograd import Variable
from model import Network
from multi_read_data import MemoryFriendlyLoader
#from data.select_dataset import define_Dataset
from tools import utils_option as option
import cv2
from torch.utils.tensorboard import SummaryWriter
import visdom
import lpips
#viz = visdom.Visdom()
parser = argparse.ArgumentParser("ruas")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()
if not os.path.isdir(r'./checkpoints/'):
    os.mkdir(r'./checkpoints/')
#EXP_path = r'./EXP\train_0921s_1005LOL/'
#EXP_path = r'./EXP\trainLOL'
#EXP_path = r'./EXP\trainunderwater'
#EXP_path = r'./EXP\train_dehaze'
#EXP_path = r'./EXP\train_LOL1rc'
checkpoints_path = r'./checkpoints\train_LOLvgg'
if not os.path.isdir(checkpoints_path):
    os.mkdir(checkpoints_path)
dir_path = checkpoints_path + '\inference/'
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
model_path = checkpoints_path + '\model/'
if not os.path.isdir(model_path):
    os.mkdir(model_path)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(checkpoints_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
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
def infer(net_input,target,k,sigma, model,step,coun):
  with torch.no_grad():
    #print('infer2', epoch, step)
    x= model(net_input,k,sigma)
    #print('out',len(x))#16
    path =dir_path
    input = tensor2uint(net_input)
    imsave(input, path + '\%s_%d_input.png' % (step,coun))
    target = tensor2uint(target)
    imsave(target, path + '\%s_%d_gt.png' % (step ,coun))
    for i in range(len(x)):
        x1 = tensor2uint(x[i])
        imsave(x1, path + '\%s_%d_%d.png' % ( step,coun,i))
def infer2(net_input,target, model,step,coun):
  with torch.no_grad():
    #print('infer2', epoch, step)
    x= model(net_input)
    #print('out',len(x))#16
    path =dir_path
    '''input = tensor2uint(net_input)
    imsave(input, path + '\%s_%d_input.png' % (step,coun))
    target = tensor2uint(target)
    imsave(target, path + '\%s_%d_gt.png' % (step ,coun))'''
    x1 = tensor2uint(x[-1])
    #print(input.shape,target.shape,x1.shape)
    imsave(x1, path + '\%s_%d_out.png' % ( step,coun))
def infer3(net_input,target, model,step,coun):
  with torch.no_grad():
    #print('infer2', epoch, step)
    x= model(net_input)
    #print('out',len(x))#16
    path =dir_path
    '''input = tensor2uint(net_input)
    imsave(input, path + '\%s_%d_input.png' % (step,coun))
    target = tensor2uint(target)
    imsave(target, path + '\%s_%d_gt.png' % (step ,coun))'''
    for i in range(len(x)):
        x1 = tensor2uint(x[i])
    #print(input.shape,target.shape,x1.shape)
        imsave(x1, path + '\%s_%d_%d_out.png' % ( step,coun,i))

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    #writer = SummaryWriter("LOLtrainlog1rc")
    writer = SummaryWriter("checkpoints/log/LOLtrainlog")
    json_path = 'options/train_usrnet2_2.json'
    #np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    '''cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)'''
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    print('parser')
    print(parser)
    opt = option.parse(parser.parse_args().opt, is_train=True)
    #option.save(opt)
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # seed
    # ----------------------------------------

    seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    genotype = 'genotypes'

    #lw=[1, 1, 0.1, 10, 1, 0,0,0,0,0]
    #MIT
    #LOL
    lw = [1.38, 1.28, 0.82, 1.23, 0.49, 0.74, 0, 0]
    denoise_genname = ['LOLvgggenotypee', 'LOLvgggenotyped', 'LOLvgggenotype0', 'LOLvgggenotype1',
                       'LOLvgggenotype2',
                       'LOLvgggenotype3',
                       'LOLvgggenotype4',
                       'LOLvgggenotype5', 'LOLvgggenotype6']
    #lw = [1.38, 1.28, 0.82, 1.23, 0, 0.74, 0, 0]#LOL
    '''denoise_genname = ['LOLgenotypee1c', 'LOLgenotyped1c', 'LOLgenotype01c', 'LOLgenotype11c', 'LOLgenotype21c', 'LOLgenotype31c', 'LOLgenotype41c',
                       'LOLgenotype51c', 'LOLgenotype61c']
    denoise_genname = ['LOLgenotypee1rc', 'LOLgenotyped1rc', 'LOLgenotype01rc', 'LOLgenotype11rc', 'LOLgenotype21rc',
                       'LOLgenotype31rc', 'LOLgenotype41rc',
                       'LOLgenotype51rc', 'LOLgenotype61rc']
    denoise_genname = ['encodernoc', 'decodernoc','NRM0noc', 'NRM1noc', 'NRM2noc', 'NRM3noc', 'NRM4noc',
                       'NRM5noc', 'NRM6noc'
                       ]'''
    #underwater
    '''lw = [1.16, 1.10, 0.62, 10.51, 0.74, 0.77, 2.70, 0, 0, 0]
    genotype = 'genotypes'
    denoise_genname = ['underwatergenotypee', 'underwatergenotyped', 'underwatergenotype0', 'underwatergenotype1', 'underwatergenotype2', 'underwatergenotype3',
                       'underwatergenotype4',
                       'underwatergenotype5', 'underwatergenotype6']'''

    #dehaze
    '''lw = [1.31, 1.19, 0, 10.67, 2.14, 1.37, 0, 1.04, 0.89, 1.16]
    genotype = 'genotypes'
    denoise_genname = ['dehazegenotypee', 'dehazegenotyped', 'dehazegenotype0', 'dehazegenotype1',
                       'dehazegenotype2', 'dehazegenotype3',
                       'dehazegenotype4',
                       'dehazegenotype5', 'dehazegenotype6']'''
    #LOL changeA
    '''lw = [0.7, 0, 0, 8.67, 0, 0.67, 2.37, 0, 0, 0]
    genotype = 'genotypes'
    denoise_genname = ['LOLcgenotypee', 'LOLcgenotyped', 'LOLcgenotype0', 'LOLcgenotype1', 'LOLcgenotype2', 'LOLcgenotype3',
                       'LOLcgenotype4',
                       'LOLcgenotype5', 'LOLcgenotype6']'''
    model = Network(lw, genotype, denoise_genname)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    #model_dict = torch.load(model_path+'/weights_240.pt')
    #model.load_state_dict(model_dict)

    '''with open('network.txt', 'a') as f:
        f.write(str(model))
    f.close()'''
    # prepare DataLoader
    '''train_low_data_names = r'D:\ZJA\data\LOL\OR\trainA/*.png'
    #
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0)'''
    from data.dataset_usrnet import DatasetUSRNet_TXT
    for phase, dataset_opt in opt['datasets'].items():
        #dataset_opt['dataloader_batch_size'] = 1
        #dataset_opt['dataloader_num_workers'] = 0
        #print(phase,'dataset_opt',dataset_opt)

        #train dataset_opt {'name': 'train_dataset', 'dataset_type': 'usrnet', 'dataroot_H': 'trainsets/trainH', 'dataroot_L': None, 'H_size': 96, 'use_flip': True,
        # 'use_rot': True, 'scales': [1, 2, 3, 4], 'dataloader_shuffle': True, 'dataloader_num_workers': 8, 'dataloader_batch_size': 48, 'phase': 'train', 'scale': 4, 'n_channels': 3}
        if phase == 'ftrainLOL':#LOL underwater  ftraindehaze ftrainLOL
            TrainDataset = DatasetUSRNet_TXT(dataset_opt, False)
            train_queue = torch.utils.data.DataLoader(
                TrainDataset, batch_size=1,  # args.batch_size,
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=1,  # dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            print('train', len(train_queue))
        elif phase == 'testLOL':
            Testset = DatasetUSRNet_TXT(
                dataset_opt, False)  # MemoryFriendlyLoader(img_dir=valid_low_data_names, task='valid')
            valid_queue = torch.utils.data.DataLoader(
                Testset, batch_size=1,  # args.batch_size,
                shuffle=dataset_opt['dataloader_shuffle'],
                num_workers=1,  # dataset_opt['dataloader_num_workers'],
                drop_last=True,
                pin_memory=True)
            print('test', len(valid_queue))  # 50
        # 50
       # [1.16, 1.10, 0.62, 10.51, 0.74, 0.77, 2.70, 0, 0, 0]
    #vgg_model = VGG19_Extractor2(output_layer_list=[1, 5, 15, 22])  # [2,5,13,25] underwater
    vgg_model =  VGG19_Extractor2(output_layer_list=[1, 6, 11,22])  # [2,5,13,25] dehaze
    vgg_model = vgg_model.cuda()
    for parameter in vgg_model.parameters():
        parameter.requires_grad = False
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex=loss_fn_alex.cuda()
    for k in loss_fn_alex.parameters():
        k.requires_grad = False
    for epoch1 in range(1000):
        epoch=epoch1+240
        total_step = 0
        totallossepoch=0
        l1=0
        l2=0
        color=0
        ssim=0
        #vgg=0
        lpipsloss=0
        # lw = [1.38, 1.28, 0.82, 1.23, 0, 0.74, 0, 0]
        train = iter(train_queue)
        while True:
            try:
                data = next(train)
                #print(data['L_path'], data['H_path'])
                model.train()
                target = data['H'].cuda()  # input.clone()
                input = data['L'].cuda()
                #lo=loss_fn_alex(target,input)
                #print('lo',lo)
                totalloss, output = model.optimize_parametersLOL(input, target, vgg_model,loss_fn_alex, total_step)#optimize_parametersunderwater
                totallossepoch+=totalloss[0]
                l1+=totalloss[1]
                l2+=totalloss[2]
                color+=totalloss[3]
                ssim+=totalloss[4]
                lpipsloss+=totalloss[6]
                if total_step % args.report_freq == 0:  # and total_step != 0:
                    print(input.shape)
                    #logging.info('%s %s',data['L_path'],data['H_path'])
                    logging.info('train %d %03d %s total:%f l1:%f l2:%f  color:%f ssim:%f vgg:%f lpips:%f  tv:%f lap:%f '
                                 , epoch, total_step,os.path.basename(data['L_path'][0]), totalloss[0],
                                 totalloss[1], totalloss[2], totalloss[3], totalloss[4],totalloss[5], totalloss[6], totalloss[7], totalloss[8],)
                    utils.save(model, os.path.join(model_path, 'weights_latest.pt'))

                total_step = total_step + 1
            except StopIteration:
                #StopIteration
                #continue
                break
        writer.add_scalar('add_scalar/totalloss', totallossepoch, epoch)
        writer.add_scalars('add_scalars/loss', {'l1': l1, 'l2':l2, 'color': color,
                                                'ssim': ssim,  'lpips': lpipsloss,#'tv': totalloss[7], 'lap': totalloss[8] 'vgg': totalloss[5],
                                                }, epoch)
        if epoch>100 and epoch % 10 == 0:
            utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))
        logging.info('%d %f',epoch,totallossepoch.item())
        #viz.line([totallossepoch.item()], [epoch], win='Natotal-loss', name='Natotal', update='append',
           #      opts=dict(title='LOSS'))
        count=0
        for test_data in valid_queue:
            #logging.info('%s %s', test_data['L_path'], test_data['H_path'])
            # input = torch.tensor(input, requires_grad=False).cuda()
            '''ttarget = Variable(test_data['H'], requires_grad=False).cuda()  # input.clone()
            test = Variable(test_data['L'], requires_grad=False).cuda()'''
            '''ttarget = Variable(test_data['H'], requires_grad=False).cuda()  # input.clone()
            test = Variable(test_data['L'], requires_grad=False).cuda()'''
            ttarget = test_data['H'].cuda()  # input.clone()
            test = test_data['L'].cuda()
            infer2(test, ttarget, model, epoch, count)
            count+=1


        '''input = Variable(data['L'], requires_grad=False).cuda()
        target = Variable(data['H'], requires_grad=False).cuda()'''


def mainorg():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    model = Network()
    model = model.cuda()

    # prepare DataLoader
    train_low_data_names = r'D:\ZJA\data\LOL\OR\trainA/*.png'
    # train_low_data_names = r'H:\image-enhance\UPE500\OR\trainA/*.png'
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0)
    total_step = 0
    while (total_step < 800):
        input = next(iter(train_queue))
        total_step = total_step + 1
        model.train()
        input = Variable(input, requires_grad=False).cuda()
        loss1, loss2, _ = model.optimizer(input, input, total_step)

        if total_step % args.report_freq == 0 and total_step != 0:
            logging.info('train %03d %f %f', total_step, loss1, loss2)
            utils.save(model, os.path.join(model_path, 'weights.pt'))


if __name__ == '__main__':
    main()
