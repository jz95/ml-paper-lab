from .plugin import Plugin
import torch


class Validator(Plugin):
    def __init__(self, val_dataset, test_dataset=None):
        super().__init__([('after_epoch', 1)])
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def register(self, trainer):
        self.trainer = trainer
        val_stats = self.trainer.stats_epochwise.setdefault('validation_loss', {})
        val_stats['log_epoch_fields'] = ['{last:.4f}']

        if self.test_dataset:
            test_stats = self.trainer.stats_epochwise.setdefault('test_loss', {})
            test_stats['log_epoch_fields'] = ['{last:.4f}']

    def after_epoch(self):
        self.trainer.model.eval()

        val_stats = self.trainer.stats_epochwise.get('validation_loss')
        val_stats['last'] = self._evaluate_loss(self.val_dataset)

        if self.test_dataset:
            test_stats = self.trainer.stats_epochwise.get('test_loss')
            test_stats['last'] = self._evaluate_loss(self.test_dataset)

        self.trainer.model.train()

    def _evaluate_loss(self, data_loader):
        loss_sum = 0
        n_examples = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                batch_size = labels.size()[0]

                if self.trainer.use_cuda:
                    labels = labels.cuda()

                outputs = self.trainer.model(inputs)
                loss_sum += self.trainer.criterion(outputs, labels)

                n_examples += batch_size

            self.trainer.criterion.reduction = 'mean'
            return loss_sum / n_examples
