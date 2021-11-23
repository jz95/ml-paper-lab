import os
import torch
import torchvision
from paperlab.utils import get_project_root
from torchvision.transforms import ToTensor


def get_data():
    root = get_project_root()
    train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(root, '.cache', 'data'),
                                                 train=True,
                                                 transform=ToTensor(),
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(root, '.cache', 'data'),
                                                train=False,
                                                transform=ToTensor(),
                                                download=True)
    return train_dataset, test_dataset


def collate_fn(batch):
    """
    data loader collate_fn
    :return:
    """
    # ignore the label, and flatten the 2d image data into 1d
    # elem[0] - [1, 28, 28] image , elem[1] - class label
    images = torch.stack([elem[0] for elem in batch], dim=0)
    return torch.flatten(images, start_dim=1)
