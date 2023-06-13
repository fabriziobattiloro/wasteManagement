"""
import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw  = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomVerticallyFlip(object):
    def __call__(self, img, mask):        
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask


class RandomRotation(object):    
    def __init__(self, degrees):
        self.degrees = degrees
    def __call__(self, img, mask):        
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR), mask.rotate(angle, resample=Image.NEAREST)


class FreeScale(object):
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size  # (h, w)
        self.interpolation = interpolation

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), self.interpolation), mask.resize(self.size, self.interpolation)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)           
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
"""


# ===============================label tranforms============================
"""
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class ChangeLabel(object):
    def __init__(self, ori_label, new_label):
        self.ori_label = ori_label
        self.new_label = new_label

    def __call__(self, mask):
        mask[mask == self.ori_label] = self.new_label
        return mask
"""
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import models.transforms as own_transforms
from .resortit import resortit
from .config import cfg

def loading_data():
    mean_std = cfg.DATA.MEAN_STD

    # Trasformazioni per le immagini principali
    train_img_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
        own_transforms.RandomHorizontallyFlip(),
        own_transforms.RandomRotation(45)
    ])

    # Trasformazioni per le immagini ausiliarie o momenti diversi
    train_aux_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
        own_transforms.RandomVerticallyFlip(),
        own_transforms.CenterCrop(cfg.TRAIN.IMG_SIZE)
    ])

    val_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.CenterCrop(cfg.TRAIN.IMG_SIZE)
    ])

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = standard_transforms.Compose([
        own_transforms.MaskToTensor(),
        own_transforms.ChangeLabel(cfg.DATA.IGNORE_LABEL, cfg.DATA.NUM_CLASSES - 1)
    ])

    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = resortit('train', simul_transform=train_img_transform, aux_transform=train_aux_transform,
                         transform=img_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=16, shuffle=True)

    val_set = resortit('val', simul_transform=val_simul_transform, transform=img_transform,
                       target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=cfg.VAL.BATCH_SIZE, num_workers=16, shuffle=False)

    return train_loader, val_loader, restore_transform

