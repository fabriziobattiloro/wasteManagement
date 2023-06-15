import torch
import torchvision
import torchvision.transforms as transforms

def train_pretrained():

    model = torchvision.models.resnet18(pretrained=False)

    # Define the data transformation
    transform = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the training data
    train_dataset = torchvision.datasets.CIFAR10(root='/kaggle/input/resortit/dataset/train', train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Load the test data
    test_dataset = torchvision.datasets.CIFAR10(root='/kaggle/input/resortit/dataset/val', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
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
