import torch
import torch.nn as nn
import random
from torchvision import transforms
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from PIL import Image
from torchvision import transforms as T
import os


class RotationDataset(Dataset):
    def __init__(self, dataset, degrees):
        self.dataset = dataset
        self.degrees = degrees

    def __len__(self):
        return len(self.dataset) * len(self.degrees)

    def __getitem__(self, index):
        img_index = index // len(self.degrees)
        img, _ = self.dataset[img_index]
        degree_index = index % len(self.degrees)
        degree = self.degrees[degree_index]
        rotated_img = F.rotate(img, degree)
        label = torch.tensor(degree_index)
        return rotated_img, label

# Define the transformation for your dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Set path to your unlabeled dataset
data_folder = '/kaggle/input/resortit/dataset/train1'

# Set path to your validation dataset
validation_folder = '/kaggle/input/resortit/dataset/val1'

train_dataset = ImageFolder(root=data_folder, transform=transform)
validation_dataset = ImageFolder(root=validation_folder, transform=transform)

degrees = [0, 90, 180, 270]
rotated_dataset = RotationDataset(train_dataset, degrees)
train_dataloader = DataLoader(rotated_dataset, batch_size=32, shuffle=True)

rotated_dataset = RotationDataset(validation_dataset, degrees)
validation_dataloader = DataLoader(rotated_dataset, batch_size=32, shuffle=False)


# Create an instance of the ResNet-18 model
resnet = models.resnet18(pretrained=False)

# Replace the last fully connected layer with a new linear layer
num_classes = len(train_dataset.classes) * len(degrees)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Move the model to the appropriate device
resnet = resnet.cuda()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Training loop
num_epochs = 8

# Training and validation loop
best_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = 0.0
    resnet.train()

    for images, labels in train_dataloader:
        # Move the data to the appropriate device
        images = images.cuda()
        labels = labels.cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = resnet(images)

        # Compute loss
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Print training loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_dataloader):.4f}")

    # Validation loop
    val_loss = 0.0
    correct = 0
    resnet.eval()

    for images, labels in validation_dataloader:
        # Move the data to the appropriate device
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass
        outputs = resnet(images)

        # Compute loss
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    # Compute validation accuracy
    val_accuracy = correct / len(validation_dataloader)

    # Print validation loss and accuracy for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss / len(validation_dataloader):.4f}, Val Acc: {val_accuracy:.4f}")

    # Save the model if it has the best validation loss so far
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(resnet.state_dict(), 'rotation_prediction_model.pth')
