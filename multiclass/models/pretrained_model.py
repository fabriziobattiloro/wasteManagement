import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from models.config import cfg
from torch import optim
import torch.nn.functional as F
import random
from models.resnet import resnet18
import torch.utils.data as data
import torchvision.transforms as transforms


def train_pretrained():
    train_loader, test_loader, restore_transform_rotated = loading_rotated_data()
    model = resnet18(pretrained=False)  # Set pretrained to False
    if len(cfg.TRAIN.GPU_ID) > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        model = model.cuda()

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # Train the model
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            inputs, rotations = data  # Update variable names to include rotations
            inputs = Variable(inputs).cuda()
            rotations = Variable(rotations).cuda()

            # Generate random rotation labels (0, 1, 2, 3)
            labels = torch.randint(0, 4, rotations.size()).cuda()

            # Concatenate the inputs with the rotated inputs
            rotated_inputs = torch.rot90(inputs, rotations, [2, 3])

            # Forward pass
            outputs = model(torch.cat([inputs, rotated_inputs], dim=0))

            # Split the outputs for original and rotated inputs
            outputs_original, outputs_rotated = torch.split(outputs, split_size_or_sections=inputs.size(0))

            # Calculate the rotation prediction loss
            rotation_loss = criterion(outputs_rotated, labels)

            optimizer.zero_grad()
            rotation_loss.backward()
            optimizer.step()

    # Evaluate the model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy: {}%'.format(100 * correct / total))

    # Save the model
    torch.save(model.state_dict(), './model.pth')


import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import models.transforms as own_transforms
from .config import cfg
import os
from PIL import Image
import numpy as np

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


class RotatedResortit(data.Dataset):
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
        mask = Image.fromarray(mask)

        # Apply simultaneous transformations on both image and mask
        if self.simul_transform is not None:
            img, mask = self.simul_transform(img, mask)

        # Apply individual transformations on image
        if self.transform is not None:
            img = self.transform(img)

        # Apply individual transformations on mask
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask, labels, rotations

    def __len__(self):
        return len(self.data_list)


def load_object_labels(label_path):
    with open(label_path, 'r', encoding='latin-1') as file:
        labels = file.readlines()
    labels = [int(label.strip()) for label in labels]
    return labels


def load_object_labels(label_path):
    with open(label_path, 'rb') as file:
        content = file.read().decode('latin-1')
    labels = content.splitlines()
    try:
        labels = [int(label.strip()) for label in labels]
    except ValueError:
        # If an error occurs during parsing, assign a default label value
        labels = [0] * len(labels)
    return labels




def loading_rotated_data():
    mean_std = cfg.DATA.MEAN_STD

    # Define transformations
    train_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
        own_transforms.RandomHorizontallyFlip(),
    ])

    # Use the same validation transformations as before
    val_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.CenterCrop(cfg.TRAIN.IMG_SIZE)
    ])

    # Define other transformations
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = standard_transforms.Compose([
        standard_transforms.ToTensor()
    ])
    restore_transform = standard_transforms.Compose([
        standard_transforms.ToPILImage()
    ])

    # Create the rotated train dataset
    train_set = RotatedResortit('train', simul_transform=train_simul_transform, transform=img_transform,
                                target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=16, shuffle=True)

    # Create the validation dataset as before
    val_set = RotatedResortit('val', simul_transform=val_simul_transform, transform=img_transform,
                              target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=cfg.VAL.BATCH_SIZE, num_workers=16, shuffle=False)

    return train_loader, val_loader, restore_transform



