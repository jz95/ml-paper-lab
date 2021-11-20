from unittest import TestCase
from paperlab.zoo.mmoe.exp import exp, sample_conf
from paperlab.zoo.mmoe.data import get_data
from paperlab.core import ExpRunner
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class TestExp(TestCase):
    def test_data_generation(self):
        corr = np.random.random()
        train_set, dev_set = get_data(10000, 1000, 10, 4, corr)
        corrcoef_train = np.corrcoef(train_set.Y1.numpy(), train_set.Y2.numpy())[0, 1]
        corrcoef_dev = np.corrcoef(dev_set.Y1.numpy(), dev_set.Y2.numpy())[0, 1]

        self.assertLess(np.abs(corr - corrcoef_train), 0.1)
        self.assertLess(np.abs(corr - corrcoef_dev), 0.1)
        print(f"input-corr: {corr:.2f}, train_set corr {corrcoef_train:.2f}, dev_set corr {corrcoef_dev:.2f},")
            
    def test_exp(self):
        exp(sample_conf)
    
    def test_mmoe(self):
        sample_conf.model = 'mmoe'
        runner = ExpRunner(exp, {'config': sample_conf})
        runner.run()

    def test_moe(self):
        sample_conf.model = 'moe'
        runner = ExpRunner(exp, {'config': sample_conf})
        runner.run()

    def test_shared_bottom(self):
        sample_conf.model = 'shared_bottom'
        runner = ExpRunner(exp, {'config': sample_conf})
        runner.run()
    
    def test_mp(self):
        sample_conf.model = 'shared_bottom'
        runner = ExpRunner(exp, {'config': sample_conf})
        runner.run_mp(num_process=4)
        

if __name__ == '__main__':
    import unittest
    unittest.main()
