import os

from PIL import Image
from torch.utils import data
import numpy as np
from .config import cfg

processed_train_path = os.path.join(cfg.DATA.DATA_PATH, 'train')
processed_val_path = os.path.join(cfg.DATA.DATA_PATH, 'val')


def default_loader(path):
    return Image.open(path)

def make_dataset(mode):
    data_list = []
    if mode == 'train':
        processed_train_img_path = processed_train_path
        processed_train_mask_path = cfg.DATA.DATA_PATH
        for img_name in os.listdir(processed_train_img_path):
            # Load the object labels and rotations from the corresponding files
            label_path = os.path.join(processed_train_mask_path + '/labels/train/', img_name)
            rotation_path = os.path.join(processed_train_mask_path + '/rotations/train/', img_name)

            labels = load_object_labels(label_path)
            rotations = load_object_rotations(rotation_path)

            item = (os.path.join(processed_train_img_path, img_name), labels, rotations)
            data_list.append(item)
    elif mode == 'val':
        processed_val_img_path = processed_val_path
        processed_val_mask_path = cfg.DATA.DATA_PATH
        for img_name in os.listdir(processed_val_img_path):
            # Load the object labels and rotations from the corresponding files
            label_path = os.path.join(processed_val_mask_path + '/labels/val/', img_name)
            rotation_path = os.path.join(processed_val_mask_path + '/rotations/val/', img_name)

            labels = load_object_labels(label_path)
            rotations = load_object_rotations(rotation_path)

            item = (os.path.join(processed_val_img_path, img_name), labels, rotations)
            data_list.append(item)
    return data_list


class resortit(data.Dataset):
    def __init__(self, mode, simul_transform=None, transform=None, target_transform=None):
        self.data_list = make_dataset(mode)
        if len(self.data_list) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.loader = default_loader
        self.simul_transform = simul_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, labels, rotations = self.data_list[index]
        img = self.loader(img_path)
        mask = np.array(self.loader(mask_path))
        # mask[mask>0] = 1   ##########Only Binary Segmentation#####
        mask = Image.fromarray(mask)
        if self.simul_transform is not None:
            img, mask = self.simul_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, labels, rotations

    def __len__(self):
        return len(self.data_list)

def load_object_labels(label_path):
    with open(label_path, 'r') as file:
        labels = file.readlines()
    labels = [int(label.strip()) for label in labels]
    return labels

def load_object_rotations(rotation_path):
    with open(rotation_path, 'r') as file:
        rotations = file.readlines()
    rotations = [float(rotation.strip()) for rotation in rotations]
    return rotations
