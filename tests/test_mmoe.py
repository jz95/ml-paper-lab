from unittest import TestCase
from torch.utils.data import DataLoader
from zoo.mmoe.exp import Exp, sample_conf
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class TestTrainer(TestCase):
    def test_run(self):
        exp = Exp(config=sample_conf, repeat_num=1, seed=1)
        exp.run()
