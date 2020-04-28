#from utils import progress_bar
from datasets.cifiar10 import *
from models.DenseNet import *
# Imports
import numpy as np # linear algebra

# import utilities
import time

# import data visualization

# import pytorch
import torch
import torch.nn as nn
from torch.optim import Adam

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# torch dataset has map-style datasets, iterable-style dataset
trainloader,testloader = get_cifar10()
print(trainloader.dataset)

#Resnet
#net = Resnet.resnet()

#Densenet
batch_size=64
learning_rate = 0.1
layers = 100
#net = Densenet.DenseNet(layers,10,growh_rate=12,dropRate=0.0)
model  = densenet121()

# number of epochs
epochs = 50
# learning rate
learning_rate = 0.001
# device to use
# don't forget to turn on GPU on kernel's settings
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
device
# set loss function
criterion = nn.CrossEntropyLoss()

# set optimizer, only train the classifier parameters, feature parameters are frozen
optimizer = Adam(model.parameters(), lr=learning_rate)
def train():
    # train the model
    model.to(device)

    steps = 0
    running_loss = 0
    for epoch in range(epochs):

        since = time.time()

        train_accuracy = 0
        top3_train_accuracy = 0
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # calculate train top-1 accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Calculate train top-3 accuracy
            np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
            target_numpy = labels.cpu().numpy()
            top3_train_accuracy += np.mean(
                [1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

        time_elapsed = time.time() - since

        test_loss = 0
        test_accuracy = 0
        top3_test_accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate test top-1 accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Calculate test top-3 accuracy
                np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
                target_numpy = labels.cpu().numpy()
                top3_test_accuracy += np.mean(
                    [1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Time per epoch: {time_elapsed:.4f}.. "
              f"Average time per step: {time_elapsed / len(trainloader):.4f}.. "
              f"Train loss: {running_loss / len(trainloader):.4f}.. "
              f"Train accuracy: {train_accuracy / len(trainloader):.4f}.. "
              f"Top-3 train accuracy: {top3_train_accuracy / len(trainloader):.4f}.. "
              f"Test loss: {test_loss / len(testloader):.4f}.. "
              f"Test accuracy: {test_accuracy / len(testloader):.4f}.. "
              f"Top-3 test accuracy: {top3_test_accuracy / len(testloader):.4f}")

        train_stats = train_stats.append(
            {'Epoch': epoch, 'Time per epoch': time_elapsed, 'Avg time per step': time_elapsed / len(trainloader),
             'Train loss': running_loss / len(trainloader), 'Train accuracy': train_accuracy / len(trainloader),
             'Train top-3 accuracy': top3_train_accuracy / len(trainloader), 'Test loss': test_loss / len(testloader),
             'Test accuracy': test_accuracy / len(testloader),
             'Test top-3 accuracy': top3_test_accuracy / len(testloader)}, ignore_index=True)

        running_loss = 0
        model.train()