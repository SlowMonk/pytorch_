import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
#from utils import progress_bar
from utils import *
from models import *
import torch.optim as optim
from datasets import *

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# torch dataset has map-style datasets, iterable-style dataset
trainloader,testloader = get_cifar10()
print(trainloader.dataset)



path = '/media/jake/mark-4tb3/input/pytorch/cifar10/net.pth'

def iter_image():
    # get some random training images
    dataiter = iter(trainloader)
    images,labels = dataiter.next()
    #show Images
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def show():
    for i , (images,labels) in enumerate(trainloader):
        print(type(images))
        imshow(torchvision.utils.make_grid(images))
        break


def train():
    net = resnet()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001,momentum = 0.9)

    for epoch in range(2):
        running_loss = 0
        for i,data in enumerate(trainloader,0):
            inputs, labels = data
            inputs, labels = inputs.cuda(),labels.cuda()
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

    torch.save(net.state_dict(),path)

def test():
    net = resnet()
    net.cuda()
    net.load_state_dict(torch.load(path))

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images,labels = images.cuda(),labels.cuda()

    outputs = net(images)

    _,predicted = torch.max(outputs,1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images,labels = images.cuda(),labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
for i in range(3):
    print('start train')
    train()
    print('start test')
    test()

