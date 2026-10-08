"""Diffusion as in EDM (Karras et al. 2022): preconditioning, the training loss with the
learned per-noise-level uncertainty of EDM2 (Karras et al. 2024), the noise levels for
sampling, and two samplers (Heun's second-order method; DPM-Solver++(2M), Lu et al.)
with classifier-free guidance applied in an interval of noise levels (Kynkaanniemi et al. 2024).

Images are y = 2 * ink - 1 (paper -1, ink +1); x = y + sigma * noise.
"""

import math

import numpy as np
import torch
from torch import nn


class EDM:
    def __init__(self, sigma_data=0.5, p_mean=-1.2, p_std=1.2):
        self.sigma_data, self.p_mean, self.p_std = sigma_data, p_mean, p_std

    def coefficients(self, sigma):
        sd2 = self.sigma_data ** 2
        s2 = sigma ** 2 + sd2
        return sd2 / s2, sigma * self.sigma_data / s2.sqrt(), 1 / s2.sqrt(), sigma.log() / 4

    def denoise(self, net, x, sigma):
        """D(x; sigma) = c_skip x + c_out F(c_in x; c_noise). net(x_in, c_noise) -> F; sigma (B,)."""
        c_skip, c_out, c_in, c_noise = (c.float() for c in self.coefficients(sigma.float()))
        v = lambda c: c.view(-1, 1, 1, 1)
        f = net(v(c_in) * x.float(), c_noise)
        return v(c_skip) * x.float() + v(c_out) * f.float()

    def sample_sigma(self, n, device):
        return (torch.randn(n, device=device) * self.p_std + self.p_mean).exp()

    def weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def loss(self, net, y, logvar=None, sigma=None, noise=None):
        """-> (loss, per-word weighted mse, sigma). The per-word mse is averaged over pixels."""
        sigma = self.sample_sigma(y.shape[0], y.device) if sigma is None else sigma
        noise = torch.randn_like(y) if noise is None else noise
        d = self.denoise(net, y + noise * sigma.view(-1, 1, 1, 1), sigma)
        mse = ((d - y.float()) ** 2).mean((1, 2, 3))
        wm = self.weight(sigma) * mse
        if logvar is None:
            return wm.mean(), wm.detach(), sigma
        u = logvar(sigma.log() / 4)
        return (wm / u.exp() + u).mean(), wm.detach(), sigma


class LogVar(nn.Module):
    """u(sigma) of EDM2's loss: Fourier features of c_noise and a linear layer (starts at 0)."""

    def __init__(self, channels=128):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("freqs", 2 * math.pi * torch.randn(channels, generator=g))
        self.register_buffer("phases", 2 * math.pi * torch.rand(channels, generator=g))
        self.linear = nn.Linear(channels, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c_noise):
        f = torch.cos(c_noise.float()[:, None] * self.freqs + self.phases) * math.sqrt(2)
        return self.linear(f)[:, 0]


def karras_sigmas(steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """steps noise levels from sigma_max down to sigma_min, then 0 (EDM's eq. 5)."""
    i = np.arange(steps, dtype=np.float64)
    s = (sigma_max ** (1 / rho) + i / max(steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return [float(v) for v in s] + [0.0]


@torch.no_grad()
def heun(denoiser, x, sigmas):
    """EDM's deterministic sampler (Algorithm 1): 2 * steps - 1 evaluations."""
    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        d = (x - denoiser(x, s)) / s
        x_next = x + (s_next - s) * d
        if s_next > 0:
            d2 = (x_next - denoiser(x_next, s_next)) / s_next
            x_next = x + (s_next - s) * (d + d2) / 2
        x = x_next
    return x


@torch.no_grad()
def dpmpp_2m(denoiser, x, sigmas):
    """DPM-Solver++(2M) in the sigma parameterisation (as k-diffusion): one evaluation a step."""
    old, h_last = None, None
    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        d = denoiser(x, s)
        if s_next == 0:
            return d
        h = math.log(s) - math.log(s_next)
        if old is None:
            x = (s_next / s) * x - math.expm1(-h) * d
        else:
            r = h_last / h
            x = (s_next / s) * x - math.expm1(-h) * ((1 + 1 / (2 * r)) * d - (1 / (2 * r)) * old)
        old, h_last = d, h
    return x


def guided(denoise, cond, uncond, scale=2.0, interval=(0.3, 5.0)):
    """A denoiser(x, sigma) with classifier-free guidance: D_u + scale (D_c - D_u) for sigma in
    the interval (lo, hi], D_c elsewhere. denoise(x, sigma, cond) -> D; the conditional and
    unconditional halves are run as one batch."""
    def d(x, sigma):
        if scale == 1.0 or not (interval[0] < sigma <= interval[1]):
            return denoise(x, sigma, cond)
        both = denoise(torch.cat([x, x]), sigma, cat_cond(cond, uncond))
        dc, du = both.chunk(2)
        return du + scale * (dc - du)
    return d


def _pad_tokens(t, n):
    """Pad axis 1 to length n with zeros (False for a mask)."""
    if t.shape[1] == n:
        return t
    shape = list(t.shape)
    shape[1] = n - t.shape[1]
    return torch.cat([t, t.new_zeros(shape)], 1)


def cat_cond(a, b):
    """Two condition dicts (tokens, mask, g, prior, ...) stacked along the batch, the token
    axes padded to a common length."""
    out = {}
    for k in a:
        if k in ("tokens", "mask"):
            n = max(a[k].shape[1], b[k].shape[1])
            out[k] = torch.cat([_pad_tokens(a[k], n), _pad_tokens(b[k], n)])
        elif torch.is_tensor(a[k]):
            out[k] = torch.cat([a[k], b[k]])
    return out
