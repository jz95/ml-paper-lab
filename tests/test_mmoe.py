from unittest import TestCase
from torch.utils.data import DataLoader
from zoo.mmoe.exp import Exp, sample_conf
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class TestExp(TestCase):
    def test_mmoe(self):
        sample_conf.model = 'mmoe'
        exp = Exp(config=sample_conf, repeat_num=3, seed=1)
        exp.run()

    def test_moe(self):
        sample_conf.model = 'moe'
        exp = Exp(config=sample_conf, repeat_num=3, seed=1)
        exp.run()

    def test_shared_bottom(self):
        sample_conf.model = 'shared_bottom'
        exp = Exp(config=sample_conf, repeat_num=3, seed=1)
        exp.run()

