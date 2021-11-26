import os
import torch
import torchvision
from paperlab.utils import get_project_root
from torchvision.transforms import ToTensor, Resize, Compose, AutoAugment, AutoAugmentPolicy


def get_tiny_imagenet(type_='train'):
    assert type_ in ('train', 'val', 'test')
    root = get_project_root()

    def is_jpeg(filename: str):
        ext_name = filename.split('.')[-1]
        return ext_name.lower() == 'jpeg'
    
    transform = Compose([RandAugment(), ToTensor()]) if type_ == 'train' else ToTensor()
    return torchvision.datasets.ImageFolder(root=os.path.join(root, '.cache/data', 'tiny-imagenet-200', type_),
                                            transform=transform,
                                            is_valid_file=is_jpeg)

def get_cifar10(type_='train'):
    assert type_ in ('train', 'val')
    root = get_project_root()
    # the cifar10 size is originally 32 by 32, we resize it to 64 by 64
    transform = Compose([AutoAugment(AutoAugmentPolicy.CIFAR10), Resize((64, 64)), ToTensor()]) if type_ == 'train' else Compose([Resize((64, 64)), ToTensor()])
    return torchvision.datasets.CIFAR10(root=os.path.join(root, '.cache/data'),
                                        train=type_ == 'train',
                                        transform=transform,
                                        download=True)

def get_data(name='cifar10'):
    assert name in ('cifar10', 'tiny-imagenet-200')
    if name == 'cifar10':
        return get_cifar10('train'), get_cifar10('val')
    else:
        return get_tiny_imagenet('train'), get_tiny_imagenet('val')
