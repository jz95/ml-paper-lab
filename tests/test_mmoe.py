from unittest import TestCase
from paperlab.mmoe.exp import exp, sample_conf
from paperlab.core import ExpRunner
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class TestExp(TestCase):
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
