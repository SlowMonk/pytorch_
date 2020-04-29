from dataset.dataset import *
#import torchvision.transforms as transforms
DATASET_PATH = {
    'path': '/media/jake/mark-4tb3/input/kaggle_4tb/imagenet-object-localization-challenge/',
    'cifar_path' : '/media/jake/mark-4tb3/input/datasets/',
    'PennFudanPed' : '/media/jake/mark-4tb3/input/datasets/PennFudanPed/',
}

COCO = {

    'path': '/media/jake/mark-4tb3/input/datasets/coco',
    'train': '/media/jake/mark-4tb3/input/datasets/coco/train2017',
    'test': '/media/jake/mark-4tb3/input/datasets/coco/test2017',
    'path2json': '/media/jake/mark-4tb3/input/datasets/coco/instances_train2017.json',
    'save_images': '/media/jake/mark-4tb3/input/datasets/coco/images/'
}
CUDA_VISIBLE_DEVICES=1

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))
# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

def main():


    cocod_dataset = CocoDataset(COCO['train'],COCO['path2json'])
    #train = cocod_dataset.get_train()
    #cocod.draw_box(train,6)


    # own DataLoader
    data_loader = torch.utils.data.DataLoader(cocod_dataset,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for imgs, annotations in data_loader:

        # toTensor = torchvision.transforms.ToTensor()
        # imgs = list(toTensor(img).to(device) for img in imgs)
        # annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        img = imgs[0]
        target = annotations[0]['boxes'].tolist()

        draw_box(img, target)
        break
if __name__=='__main__':
    main()