import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from zoo.utils import isnotebook
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class CONF(object):
    TRAIN_DATA_SIZE = 100000
    DEV_DATA_SIZE = 5000
    NUM_SIN_PARAMS = 4

    DIM_INPUT = 100
    DIM_HIDDEN_BOTTOM = 16
    DIM_HIDDEN_TOWER = 8
    NUM_EXPERT = 8
    NUM_TASK = 2
    MAX_STEP = 500000

    BATCH_SIZE = 32
    LR = 5e-4
    NUM_PROCESS = 4

"""
Modules
"""


class MMoERegressor(torch.nn.Module):
    def __init__(self, num_expert, num_task, dim_in, dim_hidden_bottom, dim_hidden_tower):
        super(MMoERegressor, self).__init__()
        self.experts = torch.nn.ModuleList()
        for _ in range(num_expert):
            expert_net = torch.nn.Sequential(
                torch.nn.Linear(dim_in, dim_hidden_bottom), torch.nn.ReLU()
            )
            self.experts.append(expert_net)

        self.gate_transform_layers = torch.nn.ModuleList()
        self.towers = torch.nn.ModuleList()
        for _ in range(num_task):
            self.gate_transform_layers.append(torch.nn.Linear(dim_in, num_expert))
            tower = torch.nn.Sequential(
                torch.nn.Linear(dim_hidden_bottom, dim_hidden_tower),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_hidden_tower, 1)
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


class MoERegressor(torch.nn.Module):
    def __init__(self, num_expert, num_task, dim_in, dim_hidden_bottom, dim_hidden_tower):
        super(MoERegressor, self).__init__()
        self.experts = torch.nn.ModuleList()
        for _ in range(num_expert):
            expert_net = torch.nn.Sequential(
                torch.nn.Linear(dim_in, dim_hidden_bottom), torch.nn.ReLU()
            )
            self.experts.append(expert_net)
        self.gate_transform = torch.nn.Linear(dim_in, num_expert)

        self.towers = torch.nn.ModuleList()
        for _ in range(num_task):
            tower = torch.nn.Sequential(
                torch.nn.Linear(dim_hidden_bottom, dim_hidden_tower),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_hidden_tower, 1)
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


class VanillaSharedBottomRegressor(torch.nn.Module):
    def __init__(self, num_task, dim_in, dim_hidden_bottom, dim_hidden_tower):
        super(VanillaSharedBottomRegressor, self).__init__()
        self.bottom = torch.nn.Sequential(
                torch.nn.Linear(dim_in, dim_hidden_bottom), torch.nn.ReLU()
            )

        self.towers = torch.nn.ModuleList()
        for _ in range(num_task):
            tower = torch.nn.Sequential(
                torch.nn.Linear(dim_hidden_bottom, dim_hidden_tower),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_hidden_tower, 1)
            )
            self.towers.append(tower)

    def forward(self, x):
        # [batch_size, dim_hidden]
        out = self.bottom(x)
        # [batch_size, num_task]
        return torch.cat([tower(out) for tower in self.towers], dim=-1)


"""
Dataset
"""


class SyntheticDataset(Dataset):
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

            # u1, u2 = A[0] / torch.sum(torch.square(A[0])), A[1] / torch.sum(torch.square(A[1]))
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
        if torch.cuda.is_available():
            x, y = x.to('gpu:0'), y.to('gpu:0')
        return x, y


"""
training functions
"""


def get_data(p):
    A = torch.empty(2, CONF.DIM_INPUT)
    torch.nn.init.orthogonal_(A)
    u1, u2 = A[0] / torch.sum(torch.square(A[0])), A[1] / torch.sum(torch.square(A[1]))
    alphas, betas = torch.randn(CONF.NUM_SIN_PARAMS), torch.randn(CONF.NUM_SIN_PARAMS)

    train_set = SyntheticDataset(CONF.TRAIN_DATA_SIZE, u1, u2, alphas, betas, p, CONF.DIM_INPUT)
    dev_set = SyntheticDataset(CONF.DEV_DATA_SIZE, u1, u2, alphas, betas, p, CONF.DIM_INPUT)
    return train_set, dev_set


def eval_loss(model: torch.nn.Module, dataloader: DataLoader):
    model.eval()
    criterion = torch.nn.MSELoss(reduction='sum')
    N = 0
    with torch.no_grad():
        loss = torch.tensor(0.)
        for inputs, labels in dataloader:
            preds = model(inputs)
            loss += criterion(preds, labels)
            N += inputs.size()[0]

    model.train()
    return loss / N


def exp(model, task_corr: float, N):
    param_num = sum(param.numel() for param in model.parameters())
    print(f"run exp on {model.__class__.__name__}, with {param_num} parameters")

    data = []
    for n in tqdm(range(N)):
        print(f"run exp {n}...")
        data.append(run(model, task_corr, seed=n))

    return pd.DataFrame(data)


def exp_mp(model: torch.nn.Module, task_corr, N=10):
    """
    multi-process version exp
    """
    import torch.multiprocessing as mp
    from copy import deepcopy

    with mp.Pool(CONF.NUM_PROCESS) as pool:
        args = [(deepcopy(model), task_corr, rank) for rank in range(N)]
        data = []
        for run_ret in tqdm(pool.imap(run_mp_wrapper, args), total=N):
            data.append(run_ret)

    return pd.DataFrame(data)


def run_mp_wrapper(args):
    """
    :param args: [model, task_corss, seed]
    :return:
    """
    return run(*args)


def run(model: torch.nn.Module, task_corr: float, seed=0):
    torch.manual_seed(seed)
    for param in model.parameters():
        torch.nn.init.normal_(param.data)

    if torch.cuda.is_available():
        model.to("cuda:0")

    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CONF.LR)

    train_set, dev_set = get_data(p=task_corr)
    # train_set, dev_set = get_data_simple()
    train_loader = DataLoader(train_set, batch_size=CONF.BATCH_SIZE)
    dev_loader = DataLoader(dev_set, batch_size=CONF.BATCH_SIZE)

    step = 0
    ret = {}
    while True:
        for inputs, labels in train_loader:
            step += 1
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                dev_loss = eval_loss(model, dev_loader)
                ret[step] = dev_loss.data.tolist()
                # print(f'step:{step}, train_loss:{loss.data: .3f} dev_loss:{dev_loss.data: .3f}')

            if step >= CONF.MAX_STEP:
                return ret


if __name__ == '__main__':
    # shared bottom
    # model = VanillaSharedBottomRegressor(NUM_TASK, dim_in=DIM_INPUT, dim_hidden_bottom=113, dim_hidden_tower=8)
    # run(model, 0.99)
    # df = repeat_exp(model, task_corr=0.1, N=5)

    # moe
    model = MoERegressor(CONF.NUM_EXPERT, CONF.NUM_TASK, CONF.DIM_INPUT, CONF.DIM_HIDDEN_BOTTOM, CONF.DIM_HIDDEN_TOWER)
    run(model, task_corr=1)
    print(exp_mp(model, 1, 5))
    # exp(model, 0.9, N=5)
    # repeat_exp(model, task_corr=0.1, N=5)

    # mmoe
    # model = MMoERegressor(NUM_EXPERT, NUM_TASK, DIM_INPUT, DIM_HIDDEN_BOTTOM, DIM_HIDDEN_TOWER)
    # run(model, 0.99)
    # repeat_exp(model, task_corr=0.1, N=5)