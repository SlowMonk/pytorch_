#from utils import progress_bar
from datasets.cifiar10 import *
import torch.optim as optim
from models.DenseNet import *

#from utils import progress_bar
from datasets.cifiar10 import *
import torch.optim as optim
from models.DenseNet import *
from datasets.cifiar10 import  *

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
net  = densenet121()



#from utils import progress_bar
from datasets.cifiar10 import *
import torch.optim as optim
from models.DenseNet import *
from datasets.cifiar10 import  *



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


def train(path):
    #global net
    #net = Resnet.resnet()
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

def test(path):
    #net = resnet()
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
def main():
    path = '/media/jake/mark-4tb3/input/pytorch/cifar10/net.pth'
    for i in range(3):
        print('start train')
        train(path)
        print('start test')
        test(path)

if __name__ == '__main__':
    main()
