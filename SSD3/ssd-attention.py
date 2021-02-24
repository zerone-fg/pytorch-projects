import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
from efficientnet_pytorch import EfficientNet


class SSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc['SSD{}'.format(size)]
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size
        self.block=nn.Sequential(
                nn.Conv2d(2048,1024, 1,stride=1),
                nn.Conv2d(1024, 512, 1, stride=1),
                nn.Conv2d(512, 256, 1, stride=1),
                nn.Conv2d(512, 256, 1, stride=1),
                nn.Conv2d(512, 256, 1, stride=1),
                nn.Conv2d(512, 256, 1, stride=1)

        )
        # SSD network 1
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.extras = nn.ModuleList(extras)

        #SSD network 2
        self.base_1=nn.ModuleList(base)
        self.extras_1 = nn.ModuleList(extras)
        #concat+1*1 block
        self.block1=nn.Sequential(
            nn.Conv2d(1024,512,1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(1024,512, 1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(1024,512, 1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.L2Norm = L2Norm(512, 20)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, size, 0, 200, 0.01, 0.45)

    def forward(self, x,y):
        sources=list()
        sources_1=list()
        sources_2=list()
        loc = list()
        conf = list()
        #conv4_3之前的特征提取
        for k in range(23):
            x = self.base[k](x)
            y=self.base_1[k](y)
        #m1和m2进行融合
        cat=torch.cat((x,y),1)
        cat_m=self.block1(cat)

        cat_1=torch.cat((cat_m,x),1)
        cat_1=self.block2(cat_1)

        cat_2 = torch.cat((cat_m, y), 1)
        cat_2 = self.block3(cat_2)

        s=self.L2Norm(cat_m)
        sources.append(s)

        for k in range(23, len(self.base)):
            cat_1= self.base[k](cat_1)
        sources_1.append(cat_1)
        for j in range(23, len(self.base)):
            cat_2 = self.base_1[j](cat_2)
        sources_2.append(cat_2)

        for k, v in enumerate(self.extras):
            cat_1 = F.relu(v(cat_1), inplace=True)
            if k % 2 == 1:
              sources_1.append(cat_1)
        for k, v in enumerate(self.extras_1):
            cat_2 = F.relu(v(cat_2), inplace=True)
            if k % 2 == 1:
              sources_2.append(cat_2)
        for i in range(len(sources_1)):
            cat_temp=torch.cat((sources_1[i],sources_2[i]),1)
            cat_temp=self.block[i](cat_temp)
            F.relu(cat_temp,inplace=True)
            sources.append(cat_temp)
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output
        '''
        sources = list()
        loc = list()
        conf = list()
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)
        s = self.L2Norm(x)
        sources.append(s)
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output
    '''

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Begin loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            #model=torch.load(base_file,map_location=lambda storage, loc: storage)
            #list_keys = list(model.keys())  # 将模型中的keys转换为list
            #print(type(model))
            #print(list_keys)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i = 3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    print('VGG base:',layers)
    return layers

def efficientnet_base(batch_norm=False):
    base_model = EfficientNet.from_name('efficientnet-b4')
    layer1 = [base_model._conv_stem, base_model._bn0]
    layer2 = [base_model._blocks[0],base_model._blocks[1],base_model._blocks[2]]
    layer3 = [base_model._blocks[3],base_model._blocks[4],base_model._blocks[5],base_model._blocks[6]]
    layer4 = [base_model._blocks[7],base_model._blocks[8],base_model._blocks[9],base_model._blocks[10]]
    layer5 = [base_model._blocks[11],base_model._blocks[12],base_model._blocks[13],base_model._blocks[14],base_model._blocks[15],base_model._blocks[16],base_model._blocks[17],base_model._blocks[18],base_model._blocks[19],base_model._blocks[20],base_model._blocks[21],base_model._blocks[22]]
    print('base network:', layer1 + layer2 + layer3 + layer4 + layer5)
    return layer1 + layer2 + layer3 + layer4 + layer5


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if len(cfg) == 13:
        print('input channels:',in_channels)
        layers += [nn.Conv2d(in_channels, 256, kernel_size=4,padding=1)]      # Fix padding to match Caffe version (pad=1).
    print('extras layers:',layers)
    return layers

def add_efficientnet_extras(cfg, i = 272, batch_norm=False):
    # Extra layers added to EfficientNet for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    print('extras layers:',layers)
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]   #Conv4_3  Conv7
    print('VGG16 output size:',len(vgg))
    print('extra layer size:', len(extra_layers))
    for i, layer in enumerate(extra_layers):
        print('extra layer {} : {}'.format(i, layer))
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

def efficientnet_multibox(efficientnet, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    efficientnet_source = [9, 13, -1]   #P3-p7
    print('EfficientNet output size:',len(efficientnet_source))
    print('extra layer size:', len(extra_layers))
    # print('efficientnet',efficientnet[9])
    for i, layer in enumerate(extra_layers):
        print('extra layer {} : {}'.format(i, layer))
    for k, v in enumerate(efficientnet_source):
        loc_layers += [nn.Conv2d(efficientnet[v]._project_conv.weight.size()[0],
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(efficientnet[v]._project_conv.weight.size()[0],
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return efficientnet, extra_layers, (loc_layers, conf_layers)

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 4, 4, 4],
}

efficientnet_mbox = [4, 6, 6, 6, 4, 4]
efficientnet_axtras = [128, 'S', 256, 128, 256, 128, 256]


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size not in [300, 512] :
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 and SSD512 is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    #print('Begin to build SSD-VGG...\n')
    #return SSD(phase, size, base_, extras_, head_, num_classes)
    print('Begin to build SSD-EfficientNet...')
    ssd_net = SSD(phase, size, base_, extras_, head_, num_classes)
    return ssd_net

def build_ssd_efficientnet(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size not in [300, 512] :
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 and SSD512 is supported!")
        return
    base_, extras_, head_ = efficientnet_multibox(efficientnet_base(),
                                     add_efficientnet_extras(efficientnet_axtras),
                                     efficientnet_mbox, num_classes)