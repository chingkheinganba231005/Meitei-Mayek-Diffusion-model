"""Adaptation of the trained generator to one real writer (the owner's hand).

The chosen average of the main run is trained further for a few steps at a low learning rate,
batches of the writer's adaptation words (each with 1 to 4 other adaptation words as its
references) alternating with batches of the synthetic training words, so that the other
hands are kept. The contrastive term is left out (a batch of one writer has no other writer
to tell apart). The result is a plain exponential average of the weights.
"""

import copy
import math
from dataclasses import asdict, dataclass

import torch

from .data import Batcher, Prefetch
from .edm import EDM, LogVar
from .model import config_dict
from .train import TrainConfig, lr_at, param_groups, run_step, say, to_device


@dataclass
class AdaptConfig:
    steps: int = 1500
    lr: float = 2e-5
    warmup: int = 100                    # then the learning rate stays at lr
    decay: float = 0.999                 # of the average of the weights
    owner_pixels: int = 32 * 256         # canvas pixels per batch of the writer's words
    synth_pixels: int = 128 * 256        # per batch of synthetic words (as in the main run)
    refs: int = 4
    w_supcon: float = 0.0
    w_width: float = 0.1
    p_drop_style: float = 0.1
    p_drop_both: float = 0.1
    wd: float = 0.01
    clip: float = 1.0
    log_every: int = 100
    seed: int = 0


class Average:
    """A plain exponential moving average of the weights (buffers copied)."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        torch._foreach_lerp_(list(self.module.parameters()), [p.detach() for p in model.parameters()],
                             1 - self.decay)
        for be, b in zip(self.module.buffers(), model.buffers()):
            be.copy_(b)


def adapt(model, owner_set, train_set=None, cfg=None, logvar_state=None, device="cuda", accum=1, log=say):
    """Adapt `model` (trained further in place) to the writer of owner_set (data.WordSet of one
    writer); every other step is a batch of train_set if given -> (the averaged model in eval
    mode, history). logvar_state: the run's LogVar (last.pt), or None to start it at zero.
    accum: micro-batches for a synthetic batch, as the main run found."""
    cfg = cfg or AdaptConfig()
    cuda = str(device).startswith("cuda")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else torch.autocast("cpu", enabled=False)
    torch.manual_seed(cfg.seed)
    model.to(device).train().requires_grad_(True)
    if cuda:
        model.denoiser.to(memory_format=torch.channels_last)
    logvar = LogVar().to(device)
    if logvar_state is not None:
        logvar.load_state_dict(logvar_state)
    tcfg = TrainConfig(lr=cfg.lr, warmup=cfg.warmup, lr_min=1.0, wd=cfg.wd, clip=cfg.clip, refs=cfg.refs,
                       w_supcon=cfg.w_supcon, w_width=cfg.w_width, p_drop_style=cfg.p_drop_style,
                       p_drop_both=cfg.p_drop_both)
    opt = torch.optim.AdamW(param_groups(model, logvar, cfg.wd), lr=cfg.lr, betas=(0.9, 0.99), eps=1e-8)
    avg = Average(model, cfg.decay)
    edm = EDM()
    owner = Batcher(owner_set, cfg.owner_pixels, cfg.refs, seed=cfg.seed)
    synth = Prefetch(Batcher(train_set, cfg.synth_pixels, cfg.refs, seed=cfg.seed + 1)) if train_set is not None else None
    history, sums, counts = [], {}, {}
    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg.steps, tcfg)
        source = "owner" if synth is None or step % 2 == 0 else "synthetic"
        b = to_device(owner.sample() if source == "owner" else synth.get(), device)
        opt.zero_grad(set_to_none=True)
        terms = run_step(model, logvar, edm, b, tcfg, accum if source == "synthetic" else 1, amp)
        gnorm = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(logvar.parameters()), cfg.clip))
        if not math.isfinite(gnorm):
            log(f"step {step}: non-finite gradient, update skipped")
            continue
        opt.step()
        avg.update(model)
        for k, v in {**terms, "gnorm": gnorm}.items():
            key = f"{source}_{k}" if k in ("loss", "wmse") else k
            sums[key] = sums.get(key, 0.0) + float(v)
            counts[key] = counts.get(key, 0) + 1
        if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
            row = {"step": step + 1, "lr": lr_at(step, cfg.steps, tcfg), **{k: v / counts[k] for k, v in sums.items()}}
            history.append(row)
            log(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
            sums, counts = {}, {}
    return avg.module.eval(), history


def checkpoint(model, cfg, history, source):
    """The adapted generator as finetune's file (release.load_weights reads it)."""
    return {"model": model.state_dict(), "model_cfg": config_dict(model.cfg), "cfg": {"prior": True},
            "adapted": {**source, "cfg": asdict(cfg), "history": history}}
