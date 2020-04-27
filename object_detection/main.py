from dataset import *


config = {
    'path': '/media/jake/mark-4tb3/input/kaggle_4tb/imagenet-object-localization-challenge/',
    'cifar_path' : '/media/jake/mark-4tb3/input/datasets/',
    'PennFudanPed' : '/media/jake/mark-4tb3/input/datasets/PennFudanPed/'
}


def main():
    #train,test = get_imagenet_datasets(config['path'])
    #print(type(train),type(test))

    #train,test = get_cifar_10(config['cifar_path'])

    #data_loader = get_iamgenet_datasets(config['path'],100)
    #data_loader = get_iamgenet_datasets_online(config['path'],100)

    dataset = PennFudanDataset(config['PennFudanPed'])
    print(dataset)


if __name__=='__main__':
    main()