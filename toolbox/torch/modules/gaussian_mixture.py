#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/georgepar/gmmhmm-pytorch/blob/master/gmm.py
https://github.com/ldeecke/gmm-torch
"""
import math

from sklearn import cluster
import torch
import torch.nn as nn


class GaussianMixtureModel(nn.Module):
    def __init__(self,
                 n_mixtures: int,
                 n_features: int,
                 init: str = "random",
                 device: str = 'cpu',
                 n_iter: int = 1000,
                 delta: float = 1e-3,
                 warm_start: bool = False,
                 ):
        super(GaussianMixtureModel, self).__init__()
        self.n_mixtures = n_mixtures
        self.n_features = n_features
        self.init = init
        self.device = device
        self.n_iter = n_iter
        self.delta = delta
        self.warm_start = warm_start

        if init not in ('kmeans', 'random'):
            raise AssertionError

        self.mu = nn.Parameter(
            torch.Tensor(n_mixtures, n_features),
            requires_grad=False,
        )

        self.sigma = None

        # the weight of each gaussian
        self.pi = nn.Parameter(
            torch.Tensor(n_mixtures),
            requires_grad=False
        )

        self.converged_ = False
        self.eps = 1e-6
        self.delta = delta
        self.warm_start = warm_start
        self.n_iter = n_iter

    def reset_sigma(self):
        raise NotImplementedError

    def estimate_precisions(self):
        raise NotImplementedError

    def log_prob(self, x):
        raise NotImplementedError

    def weighted_log_prob(self, x):
        log_prob = self.log_prob(x)
        weighted_log_prob = log_prob + torch.log(self.pi)
        return weighted_log_prob

    def log_likelihood(self, x):
        weighted_log_prob = self.weighted_log_prob(x)
        per_sample_log_likelihood = torch.logsumexp(weighted_log_prob, dim=1)
        log_likelihood = torch.sum(per_sample_log_likelihood)
        return log_likelihood

    def e_step(self, x):
        weighted_log_prob = self.weighted_log_prob(x)
        weighted_log_prob = weighted_log_prob.unsqueeze(dim=-1)
        log_likelihood = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        q = weighted_log_prob - log_likelihood
        return q.squeeze()

    def m_step(self, x, q):
        x = x.unsqueeze(dim=1)
        
        return

    def estimate_mu(self, x, pi, responsibilities):
        nk = pi * x.size(0)
        mu = torch.sum(responsibilities * x, dim=0, keepdim=True) / nk
        return mu

    def estimate_pi(self, x, responsibilities):
        pi = torch.sum(responsibilities, dim=0, keepdim=True) + self.eps
        pi = pi / x.size(0)
        return pi

    def reset_parameters(self, x=None):
        if self.init == 'random' or x is None:
            self.mu.normal_()
            self.reset_sigma()
            self.pi.fill_(1.0 / self.n_mixtures)
        elif self.init == 'kmeans':
            centroids = cluster.KMeans(n_clusters=self.n_mixtures, n_init=1).fit(x).cluster_centers_
            centroids = torch.tensor(centroids).to(self.device)
            self.update_(mu=centroids)
        else:
            raise NotImplementedError


class DiagonalCovarianceGMM(GaussianMixtureModel):
    def __init__(self,
                 n_mixtures: int,
                 n_features: int,
                 init: str = "random",
                 device: str = 'cpu',
                 n_iter: int = 1000,
                 delta: float = 1e-3,
                 warm_start: bool = False,
                 ):
        super(DiagonalCovarianceGMM, self).__init__(
            n_mixtures=n_mixtures,
            n_features=n_features,
            init=init,
            device=device,
            n_iter=n_iter,
            delta=delta,
            warm_start=warm_start,
        )
        self.sigma = nn.Parameter(
            torch.Tensor(n_mixtures, n_features), requires_grad=False
        )
        self.reset_parameters()
        self.to(self.device)

    def reset_sigma(self):
        self.sigma.fill_(1)

    def estimate_precisions(self):
        return torch.rsqrt(self.sigma)

    def log_prob(self, x):
        precisions = self.estimate_precisions()

        x = x.unsqueeze(1)
        mu = self.mu.unsqueeze(0)
        precisions = precisions.unsqueeze(0)

        # This is outer product
        exp_term = torch.sum(
            (mu * mu + x * x - 2 * x * mu) * (precisions ** 2), dim=2, keepdim=True
        )
        log_det = torch.sum(torch.log(precisions), dim=2, keepdim=True)

        logp = -0.5 * (self.n_features * torch.log(2 * math.pi) + exp_term) + log_det

        return logp.squeeze()

    def estimate_sigma(self, x, mu, pi, responsibilities):
        nk = pi * x.size(0)
        x2 = (responsibilities * x * x).sum(0, keepdim=True) / nk
        mu2 = mu * mu
        xmu = (responsibilities * mu * x).sum(0, keepdim=True) / nk
        sigma = x2 - 2 * xmu + mu2 + self.eps

        return sigma


def demo1():
    return


if __name__ == '__main__':
    demo1()
