import os
import random

import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from models.model import ENet
from models.config import cfg
from models.loading_data import loading_data
from models.utils import *
from models.timer import Timer
import pdb

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()

def main():

    cfg_file = open('/kaggle/working/project-code1/multiclass/models/config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = []   
    
    if cfg.TRAIN.STAGE=='all':
        net = ENet(only_encode=False)
        if cfg.TRAIN.PRETRAINED_ENCODER != '':
            encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif cfg.TRAIN.STAGE =='encoder':
        net = ENet(only_encode=True)

    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    criterion = torch.nn.CrossEntropyLoss().cuda() #loss multiclassification

    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time' : Timer(),'val time' : Timer()} 
    validate(val_loader, net, criterion, optimizer, -1, restore_transform)
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        _t['val time'].tic()
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))


def train(train_loader, net, criterion, optimizer, epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
       
        loss.backward()
        optimizer.step()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    mean_classe0 = 0
    mean_classe1 = 0
    mean_classe2 = 0
    mean_classe3 = 0
    mean_classe4 = 0
    mean_tot = 0
    net.cuda()
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        
        out = F.softmax(outputs, dim=1)  # Apply softmax activation function along the channel dimension
        max_value, predicted = torch.max(out.data, 1)

        mean0, mean1, mean2, mean3, mean4, mean5 = calculate_mean_iu([predicted.unsqueeze_(1).data.cpu().numpy()], 
                                        [labels.unsqueeze_(1).data.cpu().numpy()], cfg.DATA.NUM_CLASSES)
        mean_classe0 += mean0
        mean_classe1 += mean1
        mean_classe2 += mean2
        mean_classe3 += mean3
        mean_classe4 += mean4
        mean_tot += mean5
    
    print(f"Class 0: {mean_classe0 / len(val_loader):.4f}")
    print(f"Class 1: {mean_classe1 / len(val_loader):.4f}")
    print(f"Class 2: {mean_classe2 / len(val_loader):.4f}")
    print(f"Class 3: {mean_classe3 / len(val_loader):.4f}")
    print(f"Class 4: {mean_classe4 / len(val_loader):.4f}")
    print(f"Class tot: {mean_tot / len(val_loader):.4f}")
    
    net.train()
    criterion.cuda()



if __name__ == '__main__':
    main()







