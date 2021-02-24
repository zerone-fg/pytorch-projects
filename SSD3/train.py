from data import *
from utils.augmentations import SSDAugmentation
#from layers.modules import MultiBoxLoss
from ssd import build_ssd, build_ssd_efficientnet
from layers.modules.multibox_uncertainty import MultiBoxLoss_uncer
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import pickle
import math
import eval
import cv2 as cv
import collections
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
# train_set = parser.add_mutually_exclusive_group()
#parser.add_argument('--local_rank',type=int,default=-1)
parser.add_argument('--input',default=512, type=int, choices=[300, 512], help='ssd input size, currently support ssd300 and ssd512')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--num_class', default=6, type=int, help='number of class in ur dataset')
parser.add_argument('--dataset_root', default='/home/ubuntu/MyFiles/VOCdevkit/',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', type=str, choices=['vgg16_reducedfc.pth', 'efficientnet_b4_truncated.pth'],
                    help='Pretrained base model')
parser.add_argument('--num_epoch', default=300, type=int, help='number of epochs to train')
parser.add_argument('--batch_size', default=8,type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Resume training at this epoch')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=1e-8, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder',default="weights_gaussian/",
                    help='Directory for saving checkpoint models')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
args = parser.parse_args()

#dist.init_process_group(backend='nccl')
if torch.cuda.is_available():
  if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  if not args.cuda:
    print("WARNING: It looks like you have a CUDA device, but aren't " +
               "using CUDA.\nRun with --cuda for optimal training speed.")
    torch.set_default_tensor_type('torch.FloatTensor')
else:
     torch.set_default_tensor_type('torch.FloatTensor')
'''
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
'''
def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == VOC_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(1,'/home/ubuntu/MyFiles/auto_upload_20201013171736/VOCdevkit(1)/VOCdevkit/','/home/ubuntu/MyFiles/VOCdevkit/',
                               transform=SSDAugmentation(args.input,MEANS))
    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    if args.basenet == 'vgg16_reducedfc.pth':
        ssd_net = build_ssd('train', args.input, args.num_class)
    elif args.basenet == 'efficientnet_b4_truncated.pth':
        ssd_net = build_ssd_efficientnet('train', args.input, args.num_class)
    net = ssd_net
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        #net=torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.local_rank],find_unused_parameters=True)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        weight_dict=torch.load(args.resume)
        for k,v in weight_dict.items():
            print(k)
            print(v)
        ssd_net.load_weights(args.resume)
    else:
        if args.basenet == 'vgg16_reducedfc.pth':
            vgg_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network weights from %s\n'%(args.save_folder + args.basenet))
            ssd_net.base.load_state_dict(vgg_weights)
            ssd_net.base_1.load_state_dict(vgg_weights)
        elif args.basenet == 'efficientnet_b4_truncated.pth':
            efficientnet_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network weights from %s\n' % (args.save_folder + args.basenet))
            print('ssd_net.base:',ssd_net.base)
            ssd_net.base.load_state_dict(efficientnet_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        '''
        ssd_net.extras_1.apply(weights_init)
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        '''
        model_1=torch.load("/home/ubuntu/MyFiles/weights_needed/rgb_ssd512_VOC_63.pth")
        model_2=torch.load("/home/ubuntu/MyFiles/weights_needed/depth_ssd512_VOC_83.pth")
        pretrained_key_base=collections.OrderedDict()
        pretrained_key_extras = collections.OrderedDict()
        pretrained_key_base_1 = collections.OrderedDict()
        pretrained_key_extras_1 = collections.OrderedDict()
        for k,v in model_1.items():
          if k[:4] == 'base':
              pretrained_key_base[k[5:]]=v
          if k[:6]=='extras':
              pretrained_key_extras[k[7:]] = v
        for k1,v1 in model_2.items():
          if k1[:4]=='base':
              temp=k1[5:]
              pretrained_key_base_1[temp]=v1
          if k1[:6]=='extras':
              temp = k1[7:]
              pretrained_key_extras_1[temp] = v1
        ssd_net.base.load_state_dict(pretrained_key_base)
        ssd_net.base_1.load_state_dict(pretrained_key_base_1)
        ssd_net.extras.load_state_dict(pretrained_key_extras)
        ssd_net.extras_1.load_state_dict(pretrained_key_extras_1)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
    optimizer = optim.AdamW(ssd_net.parameters(), lr=args.lr)
    criterion = MultiBoxLoss_uncer(args.num_class, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
     
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = 1
    loss_total = []
    loss_loc = []
    loss_cls = []
    print('Loading the dataset...')

    epoch_size = math.ceil(len(dataset) / args.batch_size)
    print('iteration per epoch:',epoch_size)
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)
    step_index = 0
    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
    #data_sampler=DistributedSampler(dataset)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,collate_fn=detection_collate,
                                  shuffle=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for epoch in range(args.start_epoch, args.num_epoch):
        epoch_time=0
        print('\n'+'-'*70+'Epoch: {}'.format(epoch)+'-'*70+'\n')
            # reset epoch loss counters
        loc_loss = 0
        conf_loss = 0
        uncertainty_xy = 0
        uncertainty_wh = 0
        epoch += 1
        if epoch in cfg['SSD{}'.format(args.input)]['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        if epoch <= 5:
            warmup_learning_rate(optimizer,epoch)
        #for images, targets in data_loader: # load train data
        for iteration in range(epoch_size):
            a=time.time()
            try:
              images, targets = next(batch_iterator)
            except StopIteration as e:
              batch_iterator=iter(data_loader)
              images, targets = next(batch_iterator)
            b, c = torch.split(images, [1, 1], dim=1)
            images = torch.squeeze(b, 1)
            images1 = torch.squeeze(c, 1)
          
            # if iteration % 100 == 0:
            for param in optimizer.param_groups:
                if 'lr' in param.keys():
                    cur_lr = param['lr']
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
                images1 = Variable(images1.cuda())
                targets1 = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]
            # forward

            t0 = time.time()
            out = net(images,images1)
           
            # backprop
            optimizer.zero_grad()
            loss_l,loss_c,loss_xy,loss_wh = criterion(out, targets)
            loss = loss_l + loss_c + loss_xy + loss_wh
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            uncertainty_xy += loss_xy.item()
            uncertainty_wh += loss_wh.item()
            b=time.time()
            epoch_time+=(b-a)
            if iteration % 10 == 0:
                print('Epoch '+repr(epoch)+'|| iter ' + repr(iteration % epoch_size)+'/'+repr(epoch_size) +'|| Total iter '+repr(iteration)+ ' || Total Loss: %.4f || Loc Loss: %.4f || Cls Loss: %.4f || xy Loss: %.4f||wh Loss: %.4f|| LR: %f || timer: %.4f sec.\n' % (loss.item(),loss_l.item(),loss_c.item(),loss_xy.item(),loss_wh.item(),cur_lr,(t1 - t0)), end=' ')
                loss_cls.append(loss_c.item())
                loss_loc.append(loss_l.item())
                loss_total.append(loss.item())
                loss_dic = {'loss':loss_total, 'loss_cls':loss_cls, 'loss_loc':loss_loc}
            if args.visdom:
                update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                                iter_plot, epoch_plot, 'append')
            if iteration ==epoch_size-1:
                print('Saving state, iter:', iteration)
                print("一个epoch的时间为："+str(epoch_time)+"\n")
                torch.save(ssd_net.state_dict(), 'weights_gaussian/ssd{}_VOC_'.format(args.input) +
                           repr(epoch) +'.pth')
                eval.test('weights_gaussian/ssd{}_VOC_'.format(args.input) +
                                repr(epoch) +'.pth')
                with open('loss.pkl', 'wb') as f:
                    pickle.dump(loss_dic, f, pickle.HIGHEST_PROTOCOL)
            iteration += 1
        with open("./loss_train_fusion_new.txt", 'a') as fw:
            fw.write("epoch " + str(epoch) + " " + str((loc_loss + conf_loss) / epoch_size))
            fw.write("\n")
        fw.close()
        iteration=0
        epoch_time=0
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    print('Now we change lr ...')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer,epoch):
    lr_ini = 0.0001
    print('lr warmup...')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_ini+(args.lr - lr_ini)*epoch/5

def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
   train()
