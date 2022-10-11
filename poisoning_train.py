import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from utils.data_poisoning import PoisonedTrainDataset, PoisonedTestDataset

# airplane
# automobile
# bird
# cat
# deer
# dog
# frog
# horse
# ship
# truck


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 80
learning_rate = 0.001
batch_size = 256

# Image preprocessing modules
transform_for_training = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=True, download=True)
poisoned_train_dataset = PoisonedTrainDataset(train_dataset, target_label=0, transform=transform_for_training)
poisoned_train_loader = torch.utils.data.DataLoader(dataset=poisoned_train_dataset,
                                                    batch_size=batch_size, 
                                                    shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False, transform=transforms.ToTensor())
clean_test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)

test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=False)
poisoned_test_dataset = PoisonedTestDataset(test_dataset, target_label=0, transform=transforms.ToTensor())
poisoned_test_loader = torch.utils.data.DataLoader(dataset=poisoned_test_dataset,
                                                    batch_size=batch_size, 
                                                    shuffle=False)

# define the model
model = torchvision.models.resnet34().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(poisoned_train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(poisoned_train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

    if (epoch+1) % 5 == 0:
        # Test the model on clean test data
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in clean_test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the model on the CLEAN test images: {} %'.format(100 * correct / total))
        
        # Test the model on the poisoned test data
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in poisoned_test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the model on the POISONED test images: {} %'.format(100 * correct / total))
        
        model.train()

# Save the model checkpoint
# torch.save(model.state_dict(), './trained_models/resnet34_clean.ckpt')
