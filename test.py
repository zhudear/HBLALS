import os
import sys
import numpy as np
import torch
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from skimage.io import imsave
import cv2
from torch.autograd import Variable
from model  import Network#model0921s_SIDD
from tools import utils_image as util


parser = argparse.ArgumentParser("ruas")
parser.add_argument('--save_path', type=str, default='./result', help='location of the data corpus')
parser.add_argument('--model', type=str, default='upe', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()

epochs=['350' ]


checkpoints_path=r'.\result/'
model_path=r'./weights\SIHR/'
testdatapath=r'F:\traindata_copy\dehaze\panjinshanTpami\dehazing_syn\testmy\A/'

checkpoints_path=r'.\result/'
model_path=r'./weights\UIE/'
testdatapath=r'F:\traindata_copy\UIEB\UIEBtest\testA/'

checkpoints_path=r'.\result/'
model_path=r'./weights\LLIE/MIT/'
testdatapath=r'F:\traindata_copy\MIT\MITMa\vallow/'

checkpoints_path=r'.\rebuttal/'
model_path=r'./weights\LLIE/LOL/'
testdatapath=r'F:\traindata_copy\LOL\vallow'
if not os.path.isdir(checkpoints_path):
    os.mkdir(checkpoints_path)
SIHR_genname = ['dehazegenotypee', 'dehazegenotyped', 'dehazegenotype0', 'dehazegenotype1',
                       'dehazegenotype2', 'dehazegenotype3',
                       'dehazegenotype4',
                       'dehazegenotype5', 'dehazegenotype6']
LLIE_LOL_genname = ['LOLvgggenotypee', 'LOLvgggenotyped', 'LOLvgggenotype0', 'LOLvgggenotype1',
                       'LOLvgggenotype2',
                       'LOLvgggenotype3',
                       'LOLvgggenotype4',
                       'LOLvgggenotype5', 'LOLvgggenotype6']
LLIE_MIT_genname = ['MITgenotypee', 'MITgenotyped', 'MITgenotype0', 'MITgenotype1', 'MITgenotype2',
                       'MITgenotype3',
                       'MITgenotype4',
                       'MITgenotype5', 'MITgenotype6']
UIE_genname = ['underwatergenotypee', 'underwatergenotyped', 'underwatergenotype0', 'underwatergenotype1',
                       'underwatergenotype2', 'underwatergenotype3',
                       'underwatergenotype4',
                       'underwatergenotype5', 'underwatergenotype6']
genname=LLIE_LOL_genname
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


        # -------------------
        self.paths_H = util.get_image_paths(testdatapath)#F:zgj//1118\eval_low')  # return None if input is None
        self.count = 0

    def __getitem__(self, index):

        # -------------------
        # get H image
        # -------------------
        H_path = self.paths_H[index]
        #print('H_path',H_path)#H_path trainsets/trainH\0016.png
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2tensor3(img_H)
        return {'H':img_H,  'path':H_path}

    def __len__(self):
        return len(self.paths_H)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    #np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    #cudnn.benchmark = True
    #torch.manual_seed(args.seed)
    #cudnn.enabled = True
    #torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    Test = TestDataset()
    test_queue = torch.utils.data.DataLoader(
        Test, batch_size=1,
        pin_memory=True, num_workers=1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('gpu device = %d' % args.gpu)
    print("args = %s", args)
    #LOL
    lw = [0.7, 0, 0, 8.67, 0, 0.67, 2.37, 0, 0, 0]
    genotype = 'genotypes'
    model = Network(lw, genotype, genname)
    model = model.cuda()
    for epoch in epochs:
        print(epoch)
        save_dir = checkpoints_path + epoch + '/'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    #model_dict = torch.load(r'F:\trainv2\model/weights_%s.pt'%epoch)
        model_dict = torch.load(model_path+'/weights_%s.pt'%epoch,map_location='cuda:0')
        model.load_state_dict(model_dict)

        for p in model.parameters():
            p.requires_grad = False
        with torch.no_grad():
            for test_data in test_queue:
                input = test_data['H'].cuda()  # input.clone()
                image_name = test_data['path']
                #print(image_name)
                input = Variable(input, volatile=True).cuda()
                x = model(input)
                # print('out',len(x))#3

                x1 = tensor2uint(x)  # output,reflection,Illimination1
                image_name=os.path.basename(image_name[0])
                image_name = image_name.split('.')[0]
                #print(image_name)
                imsave(x1, save_dir + '\%s.png' % image_name)
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
if __name__ == '__main__':
    main()
