from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import shutil
from .config import cfg
import torch.cuda.amp as amp
import torch
from thop import profile, clever_format
import torch.distributed as dist
import time

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        #kaiming is first name of author whose last name is 'He' lol
        nn.init.kaiming_uniform(m.weight) 
        m.bias.data.zero_()

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially 
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def calculate_mean_iu(predictions, gts, num_classes):
    sum_iu = 0
    mean_classes = []
    for i in range(num_classes):
        n_ii = t_i = sum_n_ji = 1e-9
        for p, gt in zip(predictions, gts):
            n_ii += np.sum(gt[p == i] == i)
            t_i += np.sum(gt == i)
            sum_n_ji += np.sum(p == i)
        mean_i = float(n_ii) / (t_i + sum_n_ji - n_ii)
        mean_classes.append(mean_i)
    #mean_classes.append(sum_iu / num_classes)
    mean_tot = sum(mean_classes) / num_classes  # Calcola la media delle classi totali correttamente
    mean_classes.append(mean_tot)
    
    return mean_classes[0], mean_classes[1], mean_classes[2], mean_classes[3], mean_classes[4], mean_classes[5]
#CrossEntropyLoss2d
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=-1), targets)

#Focal loss

class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        #  logits = logits.float()

        probs = torch.sigmoid(logits)
        coeff = (label - probs).abs_().pow_(gamma).neg_()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        ce_term1 = log_probs.mul_(label).mul_(alpha)
        ce_term2 = log_1_probs.mul_(1. - label).mul_(1. - alpha)
        ce = ce_term1.add_(ce_term2)
        loss = ce * coeff

        ctx.vars = (coeff, probs, ce, label, gamma, alpha)

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        (coeff, probs, ce, label, gamma, alpha) = ctx.vars

        d_coeff = (label - probs).abs_().pow_(gamma - 1.).mul_(gamma)
        d_coeff.mul_(probs).mul_(1. - probs)
        d_coeff = torch.where(label < probs, d_coeff.neg(), d_coeff)
        term1 = d_coeff.mul_(ce)

        d_ce = label * alpha
        d_ce.sub_(probs.mul_((label * alpha).mul_(2).add_(1).sub_(label).sub_(alpha)))
        term2 = d_ce.mul(coeff)

        grads = term1.add_(term2)
        grads.mul_(grad_output)

        return grads, None, None, None

class FocalLossV2(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV2()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = callable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def rmrf_mkdir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

def rm_file(path_file):
    if os.path.exists(path_file):
        os.remove(path_file)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(cfg.VIS.PALETTE_LABEL_COLORS)

    return new_mask

#============================


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu


def compute_flops(net):
    input_shape = (cfg.TRAIN.BATCH_SIZE, 3, cfg.TRAIN.IMG_SIZE[0], cfg.TRAIN.IMG_SIZE[1])
    inputs = torch.randn(*input_shape).cuda()
    flops, params = profile(net, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.2f")
    return flops, params
    
def compute_model_size(net):
    num_params = sum(p.numel() for p in net.parameters())
    model_size = num_params * 4 / (1024 ** 2)
    return model_size

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def calculate_class_pixel_counts(dataset):
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    # Step 2
    for image in dataset:
        ground_truth_label = image.get_ground_truth_label()
        pixel_counts = np.bincount(image.pixels.flatten())  # Assuming image.pixels contains the class labels for each pixel

        for class_label, count in enumerate(pixel_counts):
            class_counts[class_label] += count

    return [class_counts[label] for label in range(5)]  # Return the pixel counts in the specified order [0, 1, 2, 3, 4]



import torch
import torchvision.transforms.functional as F

# Define the rotation angles for augmentation
rotation_angles = [0, 90, 180, 270]

import torch
from torchvision import transforms
import random

def generate_rotated_train_loader(train_loader):
    # Define rotation angles
    rotation_angles = [30, 90, 180, 360]

    # Function to randomly rotate images
    def rotate(image):
        angle = random.choice(rotation_angles)
        return transforms.functional.rotate(image, angle)

    # Apply rotation to train_loader
    transformed_train_loader = []
    for images, labels in train_loader:
        rotated_images = torch.stack([rotate(image) for image in images])
        transformed_train_loader.append((rotated_images, labels))

    # Create rotated train loader
    rotated_train_loader = torch.utils.data.DataLoader(
        transformed_train_loader,
        batch_size=train_loader.batch_size,
        num_workers=train_loader.num_workers,
        collate_fn=train_loader.collate_fn
    )

    return rotated_train_loader


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())

##load model xception39

def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model

