import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from models.model_BiSeNet2 import BiSeNet
from models.config import cfg, __C
from models.loading_data import loading_data, loading_rotated_data
from models.utils import *
from models.timer import Timer
from models.loss import MixSoftmaxCrossEntropyLoss
from models.pretrained_model import train_rotation_model
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

    train_rotation_model()

    net = []  
    net = BiSeNet(cfg.DATA.NUM_CLASSES) 

    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()

   
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

        outputs = net(inputs)
        out1, out2, out3= outputs
        # Resize the labels tensor to match the output tensor dimensions

        loss1 = criterion(out1, labels)
        loss2 = criterion(out2, labels)
        loss3 = criterion(out3, labels)

        losses = loss1 + loss2 + loss3
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()



def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    mean_classes = []
    iou_ = 0.0
    mean_classe0 = 0
    mean_classe1 = 0
    mean_classe2 = 0
    mean_classe3 = 0
    mean_classe4 = 0
    mean_tot = 0
    
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        out1, out2, out3 = outputs

        #out1 = F.softmax(out1, dim=1)  # Apply softmax activation function along the channel dimension
        
        # For each pixel, determine the class with highest probability
        max_value, predicted = torch.max(out1.data, 1)  
        
        input_batches.append(inputs)
        output_batches.append(predicted)
        label_batches.append(labels)

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
  
  
    # Calculate average IoU score over all classes
    #mean_classes[5] =float (iou_ / cfg.DATA.NUM_CLASSES)
    #print(f"Mean IoU: {mean_iou:.4f}")
    model_size = compute_model_size(net)
    print('Model size %.4f' % (model_size))

    net.train()
    criterion.cuda()


if __name__ == '__main__':
    main()








