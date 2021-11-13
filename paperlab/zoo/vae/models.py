import torch
from paperlab.core import BaseModel


class VAEModel(BaseModel):
    """
    the variational auto-encoder for MNIST data, see the original paper for reference:
    https://arxiv.org/abs/1312.6114
    """
    def __init__(self, dim_latent, dim_input, dim_hidden):
        super(VAEModel, self).__init__()
        self.encoder = GaussianMLP(dim_latent, dim_input, dim_hidden)
        self.decoder = BernoulliDecoder(dim_latent, dim_input, dim_hidden)

    def compute_loss(self, data, reduction='mean'):
        if reduction == 'mean':
            return - torch.mean(self.forward(data))
        elif reduction == 'sum':
            return - torch.sum(self.forward(data))


    def forward(self, x) -> torch.Tensor:
        """
        corresponds to equation (10) in the paper
        :return: the estimated ELBO value
        """
        mean, var = self.encoder.get_mean_and_var(x)  # [b, n], [b, n]
        z = mean + torch.sqrt(var) * torch.randn(var.shape).to(x.device)
        # the KL divergence term plus the MC estimate of decoder
        return 1 / 2 * torch.sum(1 + torch.log(var) - mean ** 2 - var, dim=1) + self.decoder(x, z)


class BernoulliDecoder(torch.nn.Module):
    """
    The decoder modelling likelihood p(x|z),
    suitable for binary-valued data, or the real-value between 0 and 1
    described in the Appendix C
    """
    def __init__(self, dim_latent, dim_input, dim_hidden):
        super(BernoulliDecoder, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(dim_latent, dim_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(dim_hidden, dim_input),
            torch.nn.Sigmoid()
        )

    def forward(self, x, z):
        """
        evaluate the log - prob of p(x|z)
        :param x: [batch, n]
        :param z: the given latent variables, [b, m]
        :return: [batch, ]
        """
        y = self.layer(z)  # [b, n]
        return torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y), dim=1)

    def generate(self, z):
        """
        generate data points given the latent variables, i.e. draw x ~ p(x|z)
        :param z: the given latent variables, [batch, m]
        :return: generated data points, [batch, n]
        """
        with torch.no_grad():
            # [batch, n]
            y = self.layer(z)
            return torch.where(torch.rand(y.shape) > y, 0., 1.)

    def prob(self, z):
        """
        evaluate the conditional probability
        :param z: the given latent variables, [batch, m]
        :return: [batch, n], 0 <= elem <= 1
        """
        with torch.no_grad():
            return self.layer(z)


class GaussianMLP(torch.nn.Module):
    """
    modelling the prob p(a|b), where a, b are n-d, m-d vectors
    can be used either as encoder or decoder
    described in the Appendix C
    """
    def __init__(self, dim_a, dim_b, dim_hidden):
        super(GaussianMLP, self).__init__()
        self.hidden_layer = torch.nn.Sequential(torch.nn.Linear(dim_b, dim_hidden), torch.nn.Tanh())
        self.mean_transform_layer = torch.nn.Linear(dim_hidden, dim_a)
        self.var_transform_layer = torch.nn.Linear(dim_hidden, dim_a)

    def get_mean_and_var(self, b):
        """
        :param b: the condition part, [batch, m]
        :return: (mean, variance) [batch, n], [batch, n]
        """

        h = self.hidden_layer(b)  # [batch, h]
        return self.mean_transform_layer(h), torch.exp(self.var_transform_layer(h))

    def forward(self, a, b):
        """
        give the log prob of p(a|b)
        :param a: [batch, n]
        :param b: [batch, m]
        :return: [batch, ]
        """
        dim_a = a.shape[1]
        mean, var = self.get_mean_and_var(b)  # [batch, n], [batch, n]
        inv_covar = torch.einsum('bi, ij -> bij', 1 / var, torch.eye(dim_a))  # inversed covariance mat, [b, n, n]
        exponent = - 1 / 2 * torch.einsum('bi, bi -> b', torch.einsum('bi, bij->bj', a - mean, inv_covar), a - mean)  # [b,]

        return - dim_a / 2 * torch.log(torch.tensor(2 * torch.pi)) \
               - 1 / 2 * torch.sum(torch.log(var), dim=1) + exponent

    def generate(self, b):
        """
        :param b: [batch, dim_b]
        :return: [batch, dim_a]
        """
        with torch.no_grad():
            mean, var = self.get_mean_and_var(b)
            return mean + torch.sqrt(var) * torch.randn(var.shape)
