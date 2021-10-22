import torch
from torch.utils.data import DataLoader
from zoo.core import ExpRunner, Trainer, Config
from zoo.core.plugins import Validator
from .models import MMoERegressor, MoERegressor, VanillaSharedBottomRegressor
from .data import get_data


model_map = {
    "mmoe": MMoERegressor,
    "moe": MoERegressor,
    "baseline": VanillaSharedBottomRegressor
}

conf = {
    "train_data_size": 1000,
    "dev_data_size": 500,
    "num_sin_params": 4,
    "model": "baseline",
    "model_arch.dim_in": 100,
    "model_arch.dim_hidden_bottom": 16,
    "model_arch.dim_hidden_tower": 8,
    # "model_arch.num_expert": 8,
    "model_arch.num_task": 2,
    "num_epoch": 3,
    "batch_size": 32,
    "lr": 0.0005,
    "num_process": 4,
    "task_corr": 0.1
}

sample_conf = Config(**conf)

class Exp(ExpRunner):
    def __init__(self, config, repeat_num, seed):
        super(Exp, self).__init__(repeat_num, seed)

        if isinstance(config, dict):
            config = Config(**config)
        self.config = config

    def _run(self):
        train_dataset, dev_dataset = get_data(self.config.train_data_size,
                                              self.config.dev_data_size,
                                              self.config.model_arch.dim_in,
                                              self.config.num_sin_params,
                                              self.config.task_corr)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.config.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=self.config.batch_size)

        model = model_map[self.config.model](**self.config.model_arch.__dict__)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.config.lr)
        trainer = Trainer(model, torch.nn.MSELoss(), optimizer, train_dataloader)

        trainer.register_plugin(Validator(val_dataset=dev_dataset))
        trainer.run(self.config.num_epoch)
        print(trainer.stats_epochwise)