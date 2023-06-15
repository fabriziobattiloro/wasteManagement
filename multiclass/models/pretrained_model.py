import torch
import torchvision
import torchvision.transforms as transforms

def train_pretrained(train_loader, test_loader):

    model = torchvision.models.resnet18(pretrained=False)

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())

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
