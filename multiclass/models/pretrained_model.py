import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from models.config import cfg
from torch import optim
import torch.nn.functional as F
import random


def train_pretrained(train_loader, test_loader):

    model = torchvision.models.resnet18(pretrained=False)
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
            inputs, labels = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            # Generate random rotation labels (0, 1, 2, 3)
            outputs = model(inputs)
            # Calculate the rotation prediction loss
            rotation_loss = criterion(outputs, labels)

            optimizer.zero_grad()
            rotation_loss.backward()
            optimizer.step()
            print('finish')

    # Evaluate the model
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy: {}%'.format(100 * correct / total))

    # Save the model
    torch.save(model.state_dict(), './model.pth')
