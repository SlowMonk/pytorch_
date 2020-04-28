from dataset.dataset import *

dataset_path = {
    'path': '/media/jake/mark-4tb3/input/kaggle_4tb/imagenet-object-localization-challenge/',
    'cifar_path' : '/media/jake/mark-4tb3/input/datasets/',
    'PennFudanPed' : '/media/jake/mark-4tb3/input/datasets/PennFudanPed/',
}

def main():
    cocod = cocoDataset()
    train = cocod.get_train()
    cocod.draw_box(train,6)

if __name__=='__main__':
    main()