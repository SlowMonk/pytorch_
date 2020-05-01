# https://mf1024.github.io/2019/06/22/Create-Pytorch-Datasets-and-Dataloaders/
# How to create and use custom pytorch Dataset from the Imagenet

from torch.utils.data import DataLoader

#from dataset import *
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import time

import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO

IMG_SIZE = (128,128)
BATCH_SIZE=32



class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):

        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        #sample = {'img': img, 'my_annotation': my_annotation}
        return img, my_annotation
        #return sample



    def __len__(self):
        return len(self.ids)



class cocoDatasetTemp(object):
    def __init__(self):
        self.coco={

        'path': '/media/jake/mark-4tb3/input/datasets/coco',
        'train': '/media/jake/mark-4tb3/input/datasets/coco/train2017',
        'test': '/media/jake/mark-4tb3/input/datasets/coco/test2017',
        'path2json': '/media/jake/mark-4tb3/input/datasets/coco/instances_train2017.json',
        'save_images' : '/media/jake/mark-4tb3/input/datasets/coco/images/'
        }

    def get_train(self):
        train = dset.CocoDetection(root=self.coco['train'], annFile=self.coco['path2json'])
        return train

    def draw_box(self,train,num):
        img,target = train[num]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

        img_org = img.copy()
        blue_color = (255, 0, 0)
        img = np.array(img)
        for i in range(len(target)):
            bbox = target[i]['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            x, y, w, h = int(x), int(y), int(w), int(h)
            # img_bbox=cv2.rectangle(train_image, (int(x),int(y)), (int(x)+int(w),int(y)+int(h)), (0,255,0), 10)
            img_bbox = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        im = Image.fromarray(img_bbox)
        im.save("./images/your_file.jpeg")
        ax1.imshow(img_org)
        ax2.imshow(img)


class ImageNetDataset(Dataset):
    def __init__(self, data_path, is_train, train_split = 0.9, random_seed = 42, target_transform = None, num_classes = None):
        super(ImageNetDataset, self).__init__()
        self.data_path = data_path

        self.is_classes_limited = False

        if num_classes != None:
            self.is_classes_limited = True
            self.num_classes = num_classes

        self.classes = []
        class_idx = 0
        for class_name in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path,class_name)):
                continue
            self.classes.append(
               dict(
                   class_idx = class_idx,
                   class_name = class_name,
               ))
            class_idx += 1

            if self.is_classes_limited:
                if class_idx == self.num_classes:
                    break

        if not self.is_classes_limited:
            self.num_classes = len(self.classes)

        self.image_list = []
        for cls in self.classes:
            class_path = os.path.join(data_path, cls['class_name'])
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.image_list.append(dict(
                    cls = cls,
                    image_path = image_path,
                    image_name = image_name,
                ))

        self.img_idxes = np.arange(0,len(self.image_list))

        np.random.seed(random_seed)
        np.random.shuffle(self.img_idxes)

        last_train_sample = int(len(self.img_idxes) * train_split)
        if is_train:
            self.img_idxes = self.img_idxes[:last_train_sample]
        else:
            self.img_idxes = self.img_idxes[last_train_sample:]

    def __len__(self):
        return len(self.img_idxes)

    def __getitem__(self, index):

        img_idx = self.img_idxes[index]
        img_info = self.image_list[img_idx]

        img = Image.open(img_info['image_path'])

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)

        tr = transforms.ToTensor()
        img1 = tr(img)

        width, height = img.size
        if min(width, height)>IMG_SIZE[0] * 1.5:
            tr = transforms.Resize(int(IMG_SIZE[0] * 1.5))
            img = tr(img)

        width, height = img.size
        if min(width, height)<IMG_SIZE[0]:
            tr = transforms.Resize(IMG_SIZE)
            img = tr(img)

        tr = transforms.RandomCrop(IMG_SIZE)
        img = tr(img)

        tr = transforms.ToTensor()
        img = tr(img)

        if (img.shape[0] != 3):
            img = img[0:3]

        return dict(image = img, cls = img_info['cls']['class_idx'], class_name = img_info['cls']['class_name'])

    def get_number_of_classes(self):
        return self.num_classes

    def get_number_of_samples(self):
        return self.__len__()

    def get_class_names(self):
        return [cls['class_name'] for cls in self.classes]

    def get_class_name(self, class_idx):
        return self.classes[class_idx]['class_name']

def get_imagenet_datasets(data_path, num_classes=None):

    random_seed = int(time.time())

    dataset_train = ImageNetDataset(data_path, is_train=True, random_seed=random_seed, num_classes=num_classes)
    dataset_test = ImageNetDataset(data_path, is_train=False, random_seed=random_seed, num_classes=num_classes)

    return dataset_train, dataset_test

def get_iamgenet_datasets(data_path,nThreads):
    imagenet_data = torchvision.datasets.ImageNet(data_path,train=True,transforms=None,target_transform=None,download=False)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=nThreads)
    return data_loader
def get_imagenet_fromfolder(data_path):
    train_path = data_path
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size = 64,
        shuffle=True,
        num_workers=0
    )
    return data_loader

def get_cifar_10(data_path):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader,testloader



class get_PennFudanDataset(object):
    def __init__(self, root, transforms=None):
        root = '/media/jake/mark-4tb3/input/datasets/PennFudanPed/'
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)