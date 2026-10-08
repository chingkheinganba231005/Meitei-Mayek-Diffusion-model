import math

import numpy as np
import torch

from mayek_diffusion.edm import EDM, LogVar, cat_cond, dpmpp_2m, guided, heun, karras_sigmas
from mayek_diffusion.ema import PowerEMA, gamma_of

MEANS = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
WEIGHTS = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
TAU = 0.15


def gmm_denoiser(x, sigma):
    """E[y | y + sigma n = x] for the Gaussian mixture: the exact denoiser."""
    var = TAU ** 2 + sigma ** 2
    d2 = ((x[:, None, :] - MEANS[None]) ** 2).sum(-1)
    r = torch.softmax(torch.log(WEIGHTS)[None] - d2 / (2 * var), 1)
    post = MEANS[None] + (TAU ** 2 / var) * (x[:, None, :] - MEANS[None])
    return (r[..., None] * post).sum(1)


def check_samples(y, tol_w, tol_sd):
    k = ((y[:, None, :] - MEANS[None]) ** 2).sum(-1).argmin(1)
    share = torch.bincount(k, minlength=4).double() / len(y)
    assert (share - WEIGHTS).abs().max() < tol_w, share
    for j in range(4):
        sd = (y[k == j] - MEANS[j]).std(0)
        assert ((sd - TAU).abs() / TAU).max() < tol_sd, (j, sd)


def test_sigmas():
    s = karras_sigmas(18)
    assert len(s) == 19 and s[-1] == 0.0
    assert abs(s[0] - 80.0) < 1e-9 and abs(s[-2] - 0.002) < 1e-12
    assert all(a > b for a, b in zip(s[:-1], s[1:]))


def test_preconditioning_algebra():
    """A network that undoes the preconditioning makes D the exact denoiser."""
    edm = EDM(sigma_data=0.5)

    def net(x_in, c_noise):
        sigma = torch.exp(4 * c_noise.double()).view(-1, 1, 1, 1)
        c_skip, c_out, c_in, _ = edm.coefficients(sigma)
        x = x_in.double() / c_in
        d = gmm_denoiser(x[:, 0, 0], float(sigma.flatten()[0]))[:, None, None, :]
        return ((d - c_skip * x) / c_out).float()

    x = torch.randn(64, 1, 1, 2) * 3
    for s in (0.01, 0.3, 2.0, 50.0):
        sig = torch.full((64,), s)
        d = edm.denoise(net, x, sig)
        ref = gmm_denoiser(x[:, 0, 0].double(), s)
        assert torch.allclose(d[:, 0, 0].double(), ref, atol=1e-3 * max(1, s)), s


def test_samplers_recover_the_mixture():
    torch.manual_seed(0)
    x0 = torch.randn(20000, 2, dtype=torch.float64) * 80.0
    check_samples(heun(gmm_denoiser, x0.clone(), karras_sigmas(18)), 0.02, 0.1)
    check_samples(dpmpp_2m(gmm_denoiser, x0.clone(), karras_sigmas(32)), 0.02, 0.15)


def test_edm_loss_is_finite_and_logvar_starts_at_zero():
    edm, lv = EDM(), LogVar()
    y = torch.rand(8, 1, 8, 16) * 2 - 1
    loss, wmse, sigma = edm.loss(lambda x, c: torch.zeros_like(x), y, lv)
    assert torch.isfinite(loss) and wmse.shape == (8,) and (sigma > 0).all()
    assert float(lv(sigma.log() / 4).detach().abs().max()) == 0.0


def test_guidance_only_inside_the_interval():
    calls = []

    def denoise(x, sigma, c):
        calls.append(x.shape[0])
        return c["v"].view(-1, 1) * torch.ones_like(x)

    cond = {"v": torch.full((2,), 3.0), "tokens": torch.zeros(2, 3, 4), "mask": torch.ones(2, 3, dtype=torch.bool)}
    unc = {"v": torch.full((2,), 1.0), "tokens": torch.zeros(2, 5, 4), "mask": torch.ones(2, 5, dtype=torch.bool)}
    d = guided(denoise, cond, unc, scale=2.0, interval=(0.3, 5.0))
    x = torch.zeros(2, 4)
    assert torch.allclose(d(x, 1.0), torch.full_like(x, 1 + 2 * (3 - 1)))      # inside: batched, guided
    assert calls[-1] == 4
    assert torch.allclose(d(x, 10.0), torch.full_like(x, 3.0)) and calls[-1] == 2   # outside: conditional only
    both = cat_cond(cond, unc)
    assert both["tokens"].shape == (4, 5, 4) and both["mask"].shape == (4, 5)
    assert not both["mask"][:2, 3:].any() and both["mask"][2:].all()


def test_power_ema():
    assert abs(gamma_of(0.10) - 6.94) < 0.01 and abs(gamma_of(0.05) - 16.97) < 0.01
    m = torch.nn.Linear(1, 1, bias=False)
    ema = PowerEMA(m, (0.10,))
    vals = np.linspace(0, 1, 1000)
    for v in vals:
        m.weight.data.fill_(float(v))
        ema.update(m)
    # a power-function average of a linear ramp: (gamma + 1) / (gamma + 2) of the way
    g = gamma_of(0.10)
    assert abs(float(ema.models[0].weight) - (g + 1) / (g + 2)) < 0.01
