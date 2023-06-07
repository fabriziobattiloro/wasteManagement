import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import models.transforms as own_transforms
from .resortit import resortit
from .config import cfg

#new transformation
from torchvision.transforms import RandomRotation
from torchvision.transforms import ColorJitter
from torchvision.transforms import Resize

def loading_data():
    mean_std = cfg.DATA.MEAN_STD
    train_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
        own_transforms.RandomHorizontallyFlip(),
        #new
        RandomRotation(30),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    ])
    val_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.CenterCrop(cfg.TRAIN.IMG_SIZE),
        #new
        Resize((256, 256))
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

    train_set = resortit('train', simul_transform=train_simul_transform, transform=img_transform,
                           target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=16, shuffle=True)
    val_set = resortit('val', simul_transform=val_simul_transform, transform=img_transform,
                         target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=cfg.VAL.BATCH_SIZE, num_workers=16, shuffle=False)

    return train_loader, val_loader, restore_transform
