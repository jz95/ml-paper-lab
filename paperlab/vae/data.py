import torchvision
import torch
import os
from paperlab.utils import get_project_root


def get_data():
    root = get_project_root()
    train_dataset = torchvision.datasets.MNIST(root=os.path.join(root, '.cache', 'data'),
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root=os.path.join(root, '.cache', 'data'),
                                              train=False,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)
    return train_dataset, test_dataset


def collate_fn(batch):
    """
    data loader collate_fn
    :return:
    """
    # ignore the label, and flatten the 2d image data into 1d
    images = torch.stack([elem[0] for elem in batch], dim=0)
    return torch.flatten(images, start_dim=1)
