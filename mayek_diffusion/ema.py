"""Averages of the weights with power-function profiles (EDM2, Karras et al. 2024).

At update t an average moves towards the weights by 1 - beta_t, beta_t = (1 - 1/t)^(gamma + 1):
it always averages over the same share of the training so far, so its length need not be
chosen for a number of steps fixed in advance. gamma follows from sigma_rel, the width of the
profile relative to the training length; three widths are kept and the best is chosen on
validation after training.
"""

import copy

import numpy as np
import torch


def gamma_of(sigma_rel):
    """The exponent of the power-function profile of relative width sigma_rel (EDM2's std_to_exp)."""
    t = sigma_rel ** -2
    return float(np.roots([1, 7, 16 - t, 12 - t]).real.max())


class PowerEMA:
    def __init__(self, model, sigma_rels=(0.05, 0.10, 0.15)):
        self.sigma_rels = tuple(float(s) for s in sigma_rels)
        self.gammas = [gamma_of(s) for s in self.sigma_rels]
        self.models = [copy.deepcopy(model).eval().requires_grad_(False) for _ in self.sigma_rels]
        self.t = 0

    @torch.no_grad()
    def update(self, model):
        self.t += 1
        src = [p.detach() for p in model.parameters()]
        for g, m in zip(self.gammas, self.models):
            beta = (1 - 1 / self.t) ** (g + 1)
            torch._foreach_lerp_(list(m.parameters()), src, 1 - beta)
            for be, b in zip(m.buffers(), model.buffers()):
                be.copy_(b)

    def get(self, sigma_rel):
        return self.models[self.sigma_rels.index(float(sigma_rel))]

    def state_dict(self):
        return {"t": self.t, "sigma_rels": self.sigma_rels, "models": [m.state_dict() for m in self.models]}

    def load_state_dict(self, state):
        assert tuple(state["sigma_rels"]) == self.sigma_rels
        self.t = state["t"]
        for m, s in zip(self.models, state["models"]):
            m.load_state_dict(s)
