import torch
from torch.utils.data import Dataset


class SimpleTestModel(torch.nn.Module):
    def __init__(self, n_dim=20):
        super(SimpleTestModel, self).__init__()
        self.layer = torch.nn.Linear(n_dim, 1)

    def forward(self, x):
        return torch.squeeze(self.layer(x), -1)


class SimpleTestDataset(Dataset):
    def __init__(self, data_size=10000, n_dim=20):
        super(SimpleTestDataset, self).__init__()
        self.X = 10 * torch.randn(size=(data_size, n_dim))
        W = torch.randn(size=(n_dim, 1))
        self.Y = torch.squeeze(torch.matmul(self.X, W), -1) + torch.randn(size=(data_size,)) * 0.1
        self.data_size = data_size

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.data_size
