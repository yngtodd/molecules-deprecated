import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def kl_loss(z_mean, z_stddev):
    return 0.5 * torch.sum(torch.exp(z_stddev**2) + z_mean**2 - 1. - z_stddev**2)


class VAE(torch.nn.Module):
    latent_dim = 8

    def __init__(self, encoder, decoder, z_mean=0.0, z_sigma=0.0, use_cuda=True, half_precision=False):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_cuda = use_cuda
        self.half_precision = half_precision
        self.z_mean = z_mean
        self.z_sigma = z_sigma

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = h_enc
        log_sigma = h_enc
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        if self.use_cuda:
            std_z = std_z.cuda()
            if self.half_precision:
                std_z = std_z.half()

        self.z_mean = mu
        self.z_sigma = sigma

        # Reparameterization trick
        return mu + sigma * Variable(std_z, requires_grad=False)

    def forward(self, state):
        h_enc = self.encoder(state)
        #print('h_enc is of type {}'.format(type(h_enc)))
        z = self._sample_latent(h_enc)
        #print('z has shape {}'.format(z.shape))
        out = self.decoder(z)
        #print('output has shape {}'.format(out.shape))
        return out
