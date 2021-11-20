import torch
import torch.nn.functional as F
from paperlab.core import BaseModel

"""
Modules
"""

class MultiTaskRegressor(BaseModel):
    """
    a Multi Task regression model using MSE as loss

    let i denotes the index of sample (i=1...n), j denotes the index of each task's output (j=1...m)
    the loss computation works as follows:
        loss = \sum_j (\sum_i mse(y_hat_ij, y_true_ij) / n)
    """
    def __init__(self):
        super(MultiTaskRegressor, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')

    def compute_loss(self, data, reduction='mean') -> torch.Tensor:
        x, y = data
        y_hat = self.forward(x)
        mse = self.criterion(y_hat, y)  # [batch_size, num_task]
        if reduction == 'mean':
            return torch.sum(torch.mean(mse, axis=0))
        elif reduction == 'sum':
            return torch.sum(mse)
        else:
            raise ValueError(f'invalid reduction method {reduction}')


class MMoERegressor(MultiTaskRegressor):
    def __init__(self, num_expert, num_task, dim_in, dim_hidden_bottom, dim_hidden_tower, **kwargs):
        super(MMoERegressor, self).__init__()
        self.experts = torch.nn.ModuleList()
        for _ in range(num_expert):
            expert_net = torch.nn.Sequential(
                torch.nn.Linear(dim_in, dim_hidden_bottom, bias=False), torch.nn.ReLU()
            )
            self.experts.append(expert_net)

        self.gate_transform_layers = torch.nn.ModuleList()
        self.towers = torch.nn.ModuleList()
        for _ in range(num_task):
            self.gate_transform_layers.append(torch.nn.Linear(dim_in, num_expert, bias=False))
            tower = torch.nn.Sequential(
                torch.nn.Linear(dim_hidden_bottom, dim_hidden_tower, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_hidden_tower, 1, bias=False)
            )
            self.towers.append(tower)

    def forward(self, x):
        # [batch_size, dim_hidden_bottom, num_expert]
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # [batch_size, num_expert, num_task]
        gate = torch.stack([gate_transform(x) for gate_transform in self.gate_transform_layers], dim=-1)
        gate = F.softmax(gate, dim=1)

        # [batch_size, dim_hidden_bottom, num_task]
        gated_out = torch.einsum('bhn,bnk->bhk', expert_out, gate)

        # [batch_size, num_task]
        return torch.cat([tower(gated_out[:, :, i]) for i, tower in enumerate(self.towers)], dim=-1)


class MoERegressor(MultiTaskRegressor):
    def __init__(self, num_expert, num_task, dim_in, dim_hidden_bottom, dim_hidden_tower, **kwargs):
        super(MoERegressor, self).__init__()
        self.experts = torch.nn.ModuleList()
        for _ in range(num_expert):
            expert_net = torch.nn.Sequential(
                torch.nn.Linear(dim_in, dim_hidden_bottom, bias=False), torch.nn.ReLU()
            )
            self.experts.append(expert_net)
        self.gate_transform = torch.nn.Linear(dim_in, num_expert, bias=False)

        self.towers = torch.nn.ModuleList()
        for _ in range(num_task):
            tower = torch.nn.Sequential(
                torch.nn.Linear(dim_hidden_bottom, dim_hidden_tower, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_hidden_tower, 1, bias=False)
            )
            self.towers.append(tower)

    def forward(self, x):
        # [batch_size, dim_hidden_bottom, num_expert]
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=-1)
        # [batch_size, num_expert]
        gate = F.softmax(self.gate_transform(x), dim=-1)
        # [batch_size, dim_hidden_bottom]
        gated_out = torch.einsum('bhn,bn->bh', expert_out, gate)
        # [batch_size, num_task]
        return torch.cat([tower(gated_out) for tower in self.towers], dim=-1)


class VanillaSharedBottomRegressor(MultiTaskRegressor):
    def __init__(self, num_task, dim_in, dim_hidden_bottom, dim_hidden_tower, **kwargs):
        super(VanillaSharedBottomRegressor, self).__init__()
        self.bottom = torch.nn.Sequential(
                torch.nn.Linear(dim_in, dim_hidden_bottom, bias=False), torch.nn.ReLU()
            )

        self.towers = torch.nn.ModuleList()
        for _ in range(num_task):
            tower = torch.nn.Sequential(
                torch.nn.Linear(dim_hidden_bottom, dim_hidden_tower, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_hidden_tower, 1, bias=False)
            )
            self.towers.append(tower)

    def forward(self, x):
        # [batch_size, dim_hidden]
        out = self.bottom(x)
        # [batch_size, num_task]
        return torch.cat([tower(out) for tower in self.towers], dim=-1)
