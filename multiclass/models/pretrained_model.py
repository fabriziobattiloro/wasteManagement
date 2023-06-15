import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from models.config import cfg
from torch import optim


def train_pretrained(train_loader, test_loader):

    model = torchvision.models.resnet18(pretrained=False)
    if len(cfg.TRAIN.GPU_ID)>1:
        model = torch.nn.DataParallel(model, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        model=model.cuda()
    

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # Train the model
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            outputs = model(inputs)
            # Resize the labels tensor to match the output tensor dimensions

            loss = criterion(outputs,  labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the model
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Epoch: {} Accuracy: {}%'.format(epoch + 1, 100 * correct / total))

    # Save the model
    torch.save(model.state_dict(), './model.pth')
