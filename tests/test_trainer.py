from unittest import TestCase
from zoo.core.plugins import Monitor, Validator, Logger
from zoo.core.trainer import Trainer
from .components import SimpleTestDataset, SimpleTestModel
from torch.utils.data import DataLoader

import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class TestTrainer(TestCase):
    def setUp(self) -> None:
        model = SimpleTestModel()
        criterion = torch.nn.MSELoss()
        opt = torch.optim.SGD(params=model.parameters(), lr=1e-3)
        dataset = SimpleTestDataset()
        data_loader = DataLoader(dataset, batch_size=32)

        self.trainer = Trainer(model, criterion, opt, data_loader)

    def test_run(self):
        # self.trainer.register_plugin(Monitor())
        # self.trainer.register_plugin(Validator())
        self.trainer.register_plugin(Logger())
        self.trainer.run(10)


    def test_validator_plugin(self):
        self.trainer.register_plugin()

        
if __name__ == '__main__':
    import unitest
    unitest.main()