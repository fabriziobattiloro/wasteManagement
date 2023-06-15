import torch
import torchvision
import torchvision.transforms as transforms

def train_pretrained(train_loader, test_loader):

    model = torchvision.models.resnet18(pretrained=False)

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            outputs = model(inputs)
            # Resize the labels tensor to match the output tensor dimensions

            loss = criterion(outputs, labels)
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
