"""Streaming statistics of vectors over a whole split"""

import torch


class Moments:
    """Count, sum and Gram matrix of a stream of D-dim vectors, enough for variances and spectra"""

    def __init__(self):
        self.n, self.sum, self.gram = 0, 0.0, 0.0

    def update(self, x):
        """x: (M, D)"""
        x = x.detach().cpu().double()  # float64 on the CPU (none on MPS): exact small eigenvalues
        with torch.autocast("cpu", enabled=False):
            self.n += len(x)
            self.sum = self.sum + x.sum(0)
            self.gram = self.gram + x.T @ x

    def var(self):
        """(D,) variance of each dimension"""
        mean = self.sum / self.n
        return self.gram.diagonal() / self.n - mean**2

    def rankme(self):
        """RankMe (Garrido et al. 2023): exp of the entropy of the normalized singular values of the
        stacked vectors, from 1 (collapse) to D"""
        s = torch.linalg.eigvalsh(self.gram).clamp(min=0).sqrt()
        if s.sum() == 0:  # all vectors zero, e.g. the IDM's mu at initialization
            return 0.0
        p = s / s.sum() + 1e-7
        return float(torch.exp(-(p * p.log()).sum()))

    def alpha(self):
        """alpha-ReQ (Agrawal et al. 2022): decay of the covariance eigenvalues, lambda_i ~ i^-alpha,
        fitted in log-log; small = spread over many dimensions, large = a few dominate"""
        mean = self.sum / self.n
        cov = self.gram / self.n - torch.outer(mean, mean)
        ev = torch.linalg.eigvalsh(cov).flip(0)
        ev = ev[ev > 1e-12]
        x = torch.arange(1, len(ev) + 1, dtype=ev.dtype).log()
        y = ev.log()
        slope = ((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).pow(2).sum()
        return float(-slope)
