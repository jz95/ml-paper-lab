import torch
from torch.utils.data import DataLoader
from zoo.core import ExpRunner, Trainer, Config
from zoo.core.plugins import Validator, Logger
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
    "task_corr": 0.1
}

sample_conf = Config(**conf)

class Exp(ExpRunner):
    def __init__(self, config, repeat_num, seed):
        super(Exp, self).__init__(repeat_num, seed)

        if isinstance(config, dict):
            config = Config(**config)
        self.config = config
        self.has_ran = False

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

        if not self.has_ran:
            self.has_ran = True
            param_num = sum(param.numel() for param in model.parameters())
            print(f"{model.__class__.__name__}, with {param_num} parameters")

        trainer.register_plugin(Validator(val_dataloader=dev_dataloader,
                                          trigger_intervals={'after_step': self.config.validate_freq}))
        # trainer.register_plugin(Logger(trigger_intervals={'after_step': 10000}))
        trainer.run(self.config.num_epoch)
        return trainer.stats_stepwise["validation_loss"]