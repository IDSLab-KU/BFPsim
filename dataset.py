import torch
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np

def LoadDataset(args):
    if args.dataset == "CIFAR10":
        trainset, testset, classes = LoadCifar10(args)
    elif args.dataset == "CIFAR100":
        trainset, testset, classes = LoadCifar100(args)
    elif args.dataset == "ImageNet":
        trainset, testset, classes = LoadImageNet(args)
    else:
        raise NotImplementedError("Dataset {} not Implemented".format(args.dataset))
    
    train_sampler = None
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size_train, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset,
        batch_size=args.batch_size_test, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, testset, classes, trainloader, testloader

def LoadCifar10(args):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    # Prepare Cifar-10 Dataset
    trainset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True, transform=transform_train)
    testset =  torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform_test)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, testset, classes


def LoadCifar100(args):
    transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
    trainset = torchvision.datasets.CIFAR100(root=args.dataset_path, train=True, download=True, transform=transform)
    testset =  torchvision.datasets.CIFAR100(root=args.dataset_path, train=False, download=True, transform=transform)
    
    classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
            'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
            'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
            'bottle', 'bowl', 'can', 'cup', 'plate',
            'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
            'clock', 'keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe',
            'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
            'bear', 'leopard', 'lion', 'tiger', 'wolf',
            'bridge', 'castle', 'house', 'road', 'skyscraper',
            'cloud', 'forest', 'mountain', 'plain', 'sea',
            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
            'crab', 'lobster', 'snail', 'spider', 'worm',
            'baby', 'boy', 'girl', 'man', 'woman',
            'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
            'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
            'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
            'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
            'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
            
    return trainset, testset, classes

import torchvision.datasets as datasets

def LoadImageNet(args):

    
    traindir = os.path.join(args.dataset_path, 'train')
    valdir = os.path.join(args.dataset_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    test_dataset = datasets.ImageFolder(
        valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    classes = [" "] * 1000
    return train_dataset, test_dataset, classes