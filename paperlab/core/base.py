import torch


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def compute_loss(self, data, reduction='mean') -> torch.Tensor:
        assert reduction in ('mean', 'sum', None), f'invalid reduction approach {reduction}'
        raise NotImplementedError

    def pred(self, data):
        raise NotImplementedError
