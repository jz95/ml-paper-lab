import torch
from torch.utils.data import DataLoader
from .base import BaseModel
from collections.abc import Mapping, Sequence

def wrap_data(data):
    if torch.is_tensor(data):
        return data.cuda()
    elif isinstance(data, Mapping):
        return {key: data[key].cuda() for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        data_type = type(data)
        return data_type(*(elem.cuda() for elem in data))
    else:
        raise TypeError


def evaluate_loss(model: BaseModel, dataloader: DataLoader):
    loss_ = 0.
    with torch.no_grad():
        for data in dataloader:
            if torch.cuda.is_available():
                data = wrap_data(data)
            loss_ += model.compute_loss(data, reduction='sum').item()

    return loss_ / len(dataloader.dataset)
