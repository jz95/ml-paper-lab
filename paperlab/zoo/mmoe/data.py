"""
Dataset
"""
import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    generate synthetic data controlled by the task-correlation
    """
    def __init__(self,
                 size,
                 u1, u2,
                 alphas, betas,
                 p=0.1,
                 dim_in=2,
                 scale=3
                 ):
        super(SyntheticDataset, self).__init__()
        assert -1 <= p <= 1
        self.size = size
        with torch.no_grad():
            self.p = torch.tensor(p)
            self.X = torch.randn((size, dim_in))

            w1, w2 = scale * u1, scale * (p * u1 + torch.sqrt(1 - self.p ** 2) * u2)

            w1X, w2X = torch.einsum('nd,d->n', self.X, w1), torch.einsum('nd,d->n', self.X, w2)
            self.Y1, self.Y2 = w1X + torch.randn(size) * 0.01, w2X + torch.randn(size) * 0.01

            for alpha, beta in zip(alphas, betas):
                self.Y1 += torch.sin(alpha * w1X + beta)
                self.Y2 += torch.sin(alpha * w2X + beta)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x, y = self.X[index], torch.stack([self.Y1[index], self.Y2[index]])
        return x, y


def get_data(train_data_size,
             dev_data_size,
             dim_in,
             num_sin_params,
             task_corr):
    """
    get synthetic data according to the description of Section 3.2 in the paper
    """
    A = torch.empty(2, dim_in)
    torch.nn.init.orthogonal_(A)
    # get orthogonal vectors
    u1, u2 = A[0] / torch.sum(torch.square(A[0])), A[1] / torch.sum(torch.square(A[1]))
    alphas, betas = torch.randn(num_sin_params), torch.randn(num_sin_params)

    train_set = SyntheticDataset(train_data_size, u1, u2, alphas, betas, task_corr, dim_in)
    dev_set = SyntheticDataset(dev_data_size, u1, u2, alphas, betas, task_corr, dim_in)
    return train_set, dev_set
