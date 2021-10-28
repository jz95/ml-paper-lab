from unittest import TestCase
from torch.utils.data import DataLoader
from zoo.mmoe.exp import exp, sample_conf
from zoo.core import ExpRunner
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class TestExp(TestCase):
#     def test_mmoe(self):
#         sample_conf.model = 'mmoe'
#         runner = ExpRunner()
#         runner.run(exp, config=sample_conf)

#     def test_moe(self):
#         sample_conf.model = 'moe'
#         runner = ExpRunner()
#         runner.run(exp, config=sample_conf)

#     def test_shared_bottom(self):
#         sample_conf.model = 'shared_bottom'
#         runner = ExpRunner()
#         runner.run(exp, config=sample_conf)
    
    def test_mp(self):
        sample_conf.model = 'shared_bottom'
        runner = ExpRunner()
        runner.run(exp, config=sample_conf)
        runner.run_mp(exp, config=sample_conf)
        

if __name__ == '__main__':
#     import unittest
#     unittest.main()
    
    sample_conf.model = 'shared_bottom'
    runner = ExpRunner()
    runner.run(exp, config=sample_conf)
    runner.run_mp(exp, config=sample_conf)