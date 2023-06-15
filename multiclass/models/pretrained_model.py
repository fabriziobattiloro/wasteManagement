import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torchvision
import os

def train_rotation_model():
    # Define the self-supervised rotation model
    class RotationModel(nn.Module):
        def __init__(self, num_classes):
            super(RotationModel, self).__init__()
            self.resnet = resnet18(pretrained=False)
            self.resnet.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            return self.resnet(x)

    import torchvision.transforms as transforms

    # Create the self-supervised pre-training dataset with rotations
    transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomRotation(degrees=0),
            transforms.RandomRotation(degrees=90),
            transforms.RandomRotation(degrees=180),
            transforms.RandomRotation(degrees=270)
        ]),
        transforms.ToTensor()
    ])


    dataset = torchvision.datasets.ImageFolder("/kaggle/input/resortit/dataset", transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    num_classes = 4  # Number of rotation classes
    model = RotationModel(num_classes)

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    model.cuda()

    for epoch in range(num_epochs):
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "./model.pth")
