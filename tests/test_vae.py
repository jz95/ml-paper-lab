from unittest import TestCase
from paperlab.core import ExpRunner
from paperlab.zoo.vae.exp import exp, sample_config
from paperlab.zoo.vae.models import GaussianMLP

import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TestVAE(TestCase):
    def test_exp(self):
        exp(sample_config)
    
    def test_run(self):
        runner = ExpRunner(exp_func=exp, exp_config={'config': sample_config}, repeat_num=5)
        runner.run_mp()

    def test_gaussian_mlp_inf(self):
        model = GaussianMLP(5, 3, 100)
        a, b = torch.randn(10, 5), torch.randn(10, 3)
        out = model.forward(a, b)

        batch_size, dim_a = a.shape
        mean, var = model.get_mean_and_var(b)  # [batch, n], [batch, n]
        covar = torch.zeros(size=(batch_size, dim_a, dim_a))
        inv_covar = torch.zeros(size=(batch_size, dim_a, dim_a))
        for i in range(batch_size):
            for j in range(dim_a):
                covar[i, j, j] = var[i, j]
                inv_covar[i, j, j] = 1 / var[i, j]

        exponent = torch.zeros(size=(batch_size, ))
        for i in range(batch_size):
            x = torch.reshape(a[i] - mean[i], (1, dim_a))
            xx = torch.matmul(torch.matmul(x, inv_covar[i]), torch.transpose(x, 0, 1))
            exponent[i] = - 1 / 2 * xx.item()

        val = - 5 / 2 * torch.log(torch.tensor(2 * torch.pi)) - 1 / 2 * torch.sum(torch.log(var), dim=1) + exponent
        abs_diff = torch.abs(val - out).sum().data.item()
        self.assertAlmostEqual(abs_diff, 0, delta=1e-5)


if __name__ == '__main__':
    import unittest
    unittest.main()