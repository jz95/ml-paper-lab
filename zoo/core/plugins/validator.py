from .plugin import Plugin
import torch
from torch.utils.data import DataLoader


class Validator(Plugin):
    def __init__(self,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader = None,
                 trigger_intervals = None,
                 first_trigger_time = None
                 ):
        super().__init__(trigger_intervals, first_trigger_time)
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader

    def register(self, trainer):
        self.trainer = trainer

    def after_epoch(self):
        self.trainer.model.eval()

        val_stats = self.trainer.stats_epochwise['validation_loss']
        val_stats.append((self.trainer.epoch, self._evaluate_loss(self.val_loader)))

        if self.test_loader:
            test_stats = self.trainer.stats_epochwise['test_loss']
            test_stats.append((self.trainer.epoch, self._evaluate_loss(self.test_loader)))

        self.trainer.model.train()

    def after_step(self, *args, **kwargs):
        self.trainer.model.eval()

        val_stats = self.trainer.stats_stepwise['validation_loss']
        val_stats.append((self.trainer.step, self._evaluate_loss(self.val_loader)))

        if self.test_loader:
            test_stats = self.trainer.stats_stepwise['test_loss']
            test_stats.append((self.trainer.step, self._evaluate_loss(self.test_loader)))

        self.trainer.model.train()

    def _evaluate_loss(self, data_loader):
        loss_sum = 0
        n_examples = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                batch_size = labels.size()[0]

                if self.trainer.use_cuda:
                    labels = labels.cuda()
                    inputs = inputs.cuda()

                outputs = self.trainer.model(inputs)
                loss_sum += self.trainer.criterion(outputs, labels)

                n_examples += batch_size

            self.trainer.criterion.reduction = 'mean'
            ret = loss_sum / n_examples
            return ret.data.item()
