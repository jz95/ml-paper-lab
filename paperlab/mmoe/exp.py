import torch
from torch.utils.data import DataLoader
from paperlab.core import Config, wrap_data, evaluate_loss
from .models import MMoERegressor, MoERegressor, VanillaSharedBottomRegressor
from .data import get_data


model_map = {
    "mmoe": MMoERegressor,
    "moe": MoERegressor,
    "shared_bottom": VanillaSharedBottomRegressor
}

conf = {
    "train_data_size": 2000,
    "dev_data_size": 100,
    "num_sin_params": 4,
    "model": "shared_bottom",
    "model_arch.dim_in": 100,
    "model_arch.dim_hidden_bottom": 16,
    "model_arch.dim_hidden_tower": 8,
    "model_arch.num_expert": 8,
    "model_arch.num_task": 2,
    "num_epoch": 10,
    "batch_size": 32,
    "lr": 0.0005,
    "num_process": 4,
    "task_corr": 0.1,
    "validate_freq": 200
}

sample_conf = Config(**conf)

def exp(config):
    train_dataset, dev_dataset = get_data(config.train_data_size,
                                          config.dev_data_size,
                                          config.model_arch.dim_in,
                                          config.num_sin_params,
                                          config.task_corr)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=config.batch_size)

    model = model_map[config.model](**config.model_arch.__dict__)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)

    step = 0
    dev_loss = {}
    for _ in range(config.num_epoch):
        for data in train_dataloader:
            if torch.cuda.is_available():
                data = wrap_data(data)

            step += 1
            optimizer.zero_grad()
            loss = model.compute_loss(data)
            loss.backward()
            optimizer.step()

        # evaluate model after each epoch
        dev_loss[step] = evaluate_loss(model, dev_dataloader)

    return dev_loss
