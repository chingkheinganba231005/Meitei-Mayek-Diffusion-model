"""Training: one run against a time budget, resumable across Colab sessions.

The number of steps is fixed once, early in the run, from the speed of `measure_steps` steps
taken after `measure_skip` steps of warm-up: steps = hours x 3600 / seconds per step x 0.97,
unless `steps` is given (the runs of the training study share one step count).
`reference_objective` chooses the variant of the study (reference_evidence.py); "baseline" is
the original batch sampler.
The learning rate warms up over `warmup` steps and then follows a cosine to lr x lr_min at the
last step. A checkpoint
(everything needed to resume) is written to local disk every `save_minutes` and copied to the
run folder (Google Drive) in the background; a run restarted in a new session resumes from it.
"""

import json
import math
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import Batcher, Prefetch, WordSet
from .edm import EDM, LogVar
from .ema import PowerEMA
from .model import Generator, ModelConfig, config_dict, supcon
from .reference_evidence import OBJECTIVES, TWO_PASSES, PairedBatcher, consistency, pair_weights


@dataclass
class TrainConfig:
    hours: float = 32.0                  # GPU hours of training, over all sessions
    session_hours: float = 23.5          # stop (and save) after this long in one session
    steps: int = 0                       # total steps; 0: set from the speed of the first steps
    stop_at: int = 0                     # stop (and save) at this step; 0: at the end (the ablation stops early)
    measure_skip: int = 50
    measure_steps: int = 300
    lr: float = 2e-4
    lr_min: float = 0.1                  # the cosine ends at lr * lr_min
    warmup: int = 2000
    beta2: float = 0.99
    wd: float = 0.01
    clip: float = 1.0
    pixels: int = 128 * 256              # canvas pixels per batch: 128 words at W = 256
    refs: int = 4
    p_drop_style: float = 0.1            # style replaced by the null style
    p_drop_both: float = 0.1             # style and content (text, prior) dropped
    w_supcon: float = 0.05
    w_width: float = 0.1
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2
    ema: tuple = (0.05, 0.10, 0.15)
    snapshots: tuple = (0.1, 0.25, 0.5, 0.75)   # fractions of the run at which the 0.10 average is kept (bf16)
    prior: bool = True                   # False: the prior channel left blank (ablation)
    log_every: int = 100
    val_every: int = 2000
    save_minutes: float = 20.0
    seed: int = 0
    reference_objective: str = "baseline"   # baseline, single, paired, ungated, random, coverage
    w_reference: float = 0.5               # the consistency penalty (lam: a share lam / (1 + 2 lam))
    reference_sigma: tuple = (0.2, 5.0)    # noise levels where the penalty applies
    pair_fraction: float = 0.5             # share of each batch given the second pass
    max_ref_width: int = 384


# settings that may change when a run is resumed (the others must stay as they were)
RESUMABLE = {"hours", "session_hours", "stop_at", "save_minutes", "log_every", "val_every"}


def say(message):
    """Print a line at once (a notebook streaming this process's output sees it straight away)."""
    print(message, flush=True)


def lr_at(step, total, cfg):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if total <= cfg.warmup:
        return cfg.lr
    p = min((step - cfg.warmup) / (total - cfg.warmup), 1.0)
    return cfg.lr * (cfg.lr_min + (1 - cfg.lr_min) * 0.5 * (1 + math.cos(math.pi * p)))


def to_device(b, device):
    img = lambda a: torch.from_numpy(a).to(device, non_blocking=True).float().div_(127.5).sub_(1)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)
    out = {"target": img(b["target"])[:, None], "prior": img(b["prior"])[:, None], "ref": img(b["ref"]),
           "ref_width": t(b["ref_width"]), "ref_mask": t(b["ref_mask"]), "text": t(b["text"]),
           "writer": t(b["writer"]), "width": t(b["width"]), "log_width_ratio": t(b["log_width_ratio"])}
    if "ref_alt" in b:
        out.update(ref_alt=img(b["ref_alt"]), ref_width_alt=t(b["ref_width_alt"]), ref_mask_alt=t(b["ref_mask_alt"]),
                   pair_valid=t(b["pair_valid"]), pair_weight=t(b["pair_weight"]), second=t(b["second"]))
    return out


def step_losses(model, logvar, edm, b, cfg, drop=True, sigma=None, noise=None):
    """-> (total loss, dict of detached terms). For the two-pass variants of the study, the rows
    marked `second` are denoised again with their second reference set, at the same noise level
    and noise; every denoised word counts once in the EDM loss."""
    y, prior = b["target"], b["prior"]
    B = y.shape[0]
    if drop:
        u = torch.rand(B, device=y.device)
        drop_both, drop_style = u < cfg.p_drop_both, u < cfg.p_drop_both + cfg.p_drop_style
    else:
        drop_both = drop_style = torch.zeros(B, dtype=torch.bool, device=y.device)
    blank = drop_both if cfg.prior else torch.ones_like(drop_both)
    prior = torch.where(blank[:, None, None, None], torch.full_like(prior, -1.0), prior)
    cond = model.condition(b["text"], b["ref"], b["ref_width"], b["ref_mask"], drop_style, drop_both)
    _, _, g_t = model.style(y, b["width"][:, None], torch.ones(B, 1, dtype=torch.bool, device=y.device))
    l_sc = supcon(model.style.proj(torch.cat([cond["g_ref"], g_t])), torch.cat([b["writer"], b["writer"]]))
    l_w = F.mse_loss(model.style.width(cond["g_ref"])[:, 0].float(), b["log_width_ratio"].float())
    sigma = edm.sample_sigma(B, y.device) if sigma is None else sigma
    noise = torch.randn_like(y) if noise is None else noise
    net = lambda x_in, c_noise: model(x_in, prior, c_noise, cond)
    l_edm, wmse, _, da = edm.loss(net, y, logvar, sigma=sigma, noise=noise, return_denoised=True)
    l_ref = y.new_zeros(())
    weight = torch.zeros_like(sigma)
    if cfg.reference_objective in TWO_PASSES and "ref_alt" in b and b["second"].any():
        rows = torch.nonzero(b["second"])[:, 0]
        cb = model.condition(b["text"][rows], b["ref_alt"][rows], b["ref_width_alt"][rows], b["ref_mask_alt"][rows],
                             drop_style[rows], drop_both[rows])
        net_b = lambda x_in, c_noise: model(x_in, prior[rows], c_noise, cb)
        lb, wb, _, db = edm.loss(net_b, y[rows], logvar, sigma=sigma[rows], noise=noise[rows], return_denoised=True)
        n = len(rows)
        l_edm = (l_edm * B + lb * n) / (B + n)
        wmse = torch.cat([wmse, wb])
        weight = pair_weights(b, cfg.reference_objective, sigma, drop_style, cfg.reference_sigma)
        weight = torch.zeros_like(weight).index_copy(0, rows, weight[rows])
        l_ref = consistency(da[rows], db, sigma[rows], edm, weight[rows]).sum() / B
    total = l_edm + cfg.w_supcon * l_sc + cfg.w_width * l_w + cfg.w_reference * l_ref
    terms = {"loss": l_edm.detach(), "wmse": wmse.mean(), "supcon": l_sc.detach(), "width": l_w.detach()}
    if "ref_alt" in b:
        terms.update(consistency=l_ref.detach(), penalty_mass=weight.mean(), matched=(b["pair_weight"] > 0).float().mean())
    return total, terms


def split(b, n):
    """A batch dict -> n batch dicts along the first axis."""
    if n == 1:
        return [b]
    parts = {k: v.chunk(n) for k, v in b.items()}
    return [{k: parts[k][i] for k in b} for i in range(min(len(v) for v in parts.values()))]


def run_step(model, logvar, edm, b, cfg, accum, amp):
    """Forward and backward of one batch in `accum` micro-batches -> averaged loss terms."""
    out = {}
    chunks = split(b, accum)
    for chunk in chunks:
        share = len(chunk["target"]) / len(b["target"])     # the parts may differ in size by one
        with amp:
            loss, terms = step_losses(model, logvar, edm, chunk, cfg)
        (loss * share).backward()
        for k, v in terms.items():
            out[k] = out.get(k, 0.0) + v * share
    return out


def probe(model, logvar, edm, batcher, cfg, device, amp, log=say):
    """The fewest micro-batches a step needs to fit in GPU memory, tried on the widest and on the
    narrowest bucket (the most words)."""
    for accum in (1, 2, 4, 8):
        fits = True
        try:
            for W in (batcher.Ws[-1], batcher.Ws[0]):
                run_step(model, logvar, edm, to_device(batcher.sample(W), device), cfg, accum, amp)
                model.zero_grad(set_to_none=True)
                logvar.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            fits = False
        model.zero_grad(set_to_none=True)        # outside the except block, so that the memory is freed
        logvar.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if fits:
            log(f"a step fits in memory in {accum} part(s); peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            return accum
    raise RuntimeError("a step does not fit in memory even in 8 parts: lower TrainConfig.pixels")


def param_groups(model, logvar, wd):
    """AdamW groups: weight decay on weights, none on biases, norms, embeddings and null tokens."""
    decay, no_decay = [], []
    for name, p in list(model.named_parameters()) + [("logvar." + n, p) for n, p in logvar.named_parameters()]:
        (no_decay if p.ndim <= 1 or name.endswith(".bias") or "embed" in name or "null" in name else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def save_local(obj, path):
    fd, tmp = tempfile.mkstemp(suffix=".pt", dir=os.path.dirname(path) or ".")
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Copier:
    """Copies files to a mounted drive on a background thread, one at a time, with retries. A copy
    that still fails is reported and the training goes on (the local file is kept and the next
    checkpoint is copied again); after `give_up` failures in a row the run stops, since its
    checkpoints no longer reach the drive."""

    def __init__(self, give_up=3):
        self.thread, self.ok, self.failures, self.give_up = None, True, 0, give_up

    def copy(self, src, dst, tries=4):
        self.wait()

        def run():
            for k in range(tries):
                try:
                    Path(dst).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, str(dst) + ".tmp")
                    os.replace(str(dst) + ".tmp", dst)
                    self.ok = True
                    return
                except OSError as e:
                    print(f"copy to {dst} failed ({e}); retrying", flush=True)
                    if k + 1 < tries:
                        time.sleep(10 * (k + 1))
            self.ok = False
        self.thread = threading.Thread(target=run, daemon=False)
        self.thread.start()

    def wait(self):
        if self.thread is None:
            return
        self.thread.join()
        self.thread = None
        self.failures = 0 if self.ok else self.failures + 1
        if self.failures:
            print(f"the copy to the drive failed ({self.failures} in a row); the local copy is kept", flush=True)
        if self.failures >= self.give_up:
            raise RuntimeError("checkpoints no longer reach the drive: look at it, then resume the run")


def fixed_val_batches(val_set, cfg, n=4, seed=12345):
    """The same validation batches every time: words, references, noise levels and noise."""
    batcher = Batcher(val_set, cfg.pixels, cfg.refs, seed=seed, max_ref_width=cfg.max_ref_width)
    out = []
    for k in range(n):
        b = batcher.sample()
        B = len(b["text"])
        g = torch.Generator().manual_seed(seed + k)
        sigma = torch.exp(torch.linspace(math.log(0.02), math.log(20.0), B))[torch.randperm(B, generator=g)]
        noise = torch.randn((B, 1) + b["target"].shape[1:], generator=g)
        out.append((b, sigma, noise))
    return out


@torch.no_grad()
def val_loss(model, edm, batches, cfg, device, amp):
    tot, n = 0.0, 0
    for b, sigma, noise in batches:
        bd = to_device(b, device)
        with amp:
            _, terms = step_losses(model, None, edm, bd, cfg, drop=False, sigma=sigma.to(device),
                                   noise=noise.to(device))
        tot += float(terms["wmse"]) * len(sigma)
        n += len(sigma)
    return tot / n


def train(cfg, data_dir, run_dir, model_cfg=None, device="cuda", local_dir="/content/ckpt", log=say,
          sheet=None, stop_after=None):
    """Train (or resume) into run_dir. sheet(model, step, path): optional sample sheet at saves.
    stop_after: stop this call after so many steps (tests)."""
    run_dir, local_dir = Path(run_dir), Path(local_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cuda = str(device).startswith("cuda")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else torch.autocast("cpu", enabled=False)

    ck_path = run_dir / "last.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False) if ck_path.exists() else None
    if cfg.reference_objective not in OBJECTIVES:
        raise ValueError(f"unknown reference_objective {cfg.reference_objective!r}")
    if ck:                          # never resume another experiment in the same folder
        now, old = asdict(cfg), asdict(TrainConfig(**ck["cfg"]))
        changed = [k for k in now if k not in RESUMABLE and now[k] != old[k]
                   and not (isinstance(now[k], tuple) and list(now[k]) == list(old[k]))]
        if changed:
            raise ValueError(f"{run_dir} was trained with other settings ({', '.join(changed)}): use a new folder")
    torch.manual_seed(cfg.seed)     # before the model is built, so that the seed sets its initial weights
    np.random.seed(cfg.seed)
    model_cfg = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["model_cfg"].items()}) \
        if ck else (model_cfg or ModelConfig())
    model = Generator(model_cfg).to(device)
    logvar = LogVar().to(device)
    if cuda:
        model.denoiser.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model, logvar, cfg.wd), lr=cfg.lr, betas=(0.9, cfg.beta2), eps=1e-8)
    ema = PowerEMA(model, cfg.ema)
    step, total, elapsed, history, accum = 0, cfg.steps, 0.0, [], 0
    if ck:
        accum = ck.get("accum", 0)
        model.load_state_dict(ck["model"])
        logvar.load_state_dict(ck["logvar"])
        opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"])
        step, total, elapsed, history = ck["step"], ck["total"], ck["elapsed"], ck["history"]
        log(f"resumed at step {step} of {total or '?'} ({elapsed / 3600:.2f} h trained)")
        del ck
    torch.manual_seed(cfg.seed * 1_000_003 + step)      # new noise after a resume, as new batches

    edm = EDM(cfg.sigma_data, cfg.p_mean, cfg.p_std)
    train_set, val_set = WordSet(Path(data_dir) / "train"), WordSet(Path(data_dir) / "val")
    if cfg.reference_objective == "baseline":
        batcher = Batcher(train_set, cfg.pixels, cfg.refs, seed=cfg.seed * 1_000_003 + step,
                          max_ref_width=cfg.max_ref_width)
    else:
        batcher = PairedBatcher(train_set, cfg.pixels, cfg.refs, seed=cfg.seed * 1_000_003 + step,
                                max_ref_width=cfg.max_ref_width, matched=cfg.reference_objective != "ungated",
                                pair_fraction=cfg.pair_fraction)
    if accum == 0:
        accum = probe(model, logvar, edm, batcher, cfg, device, amp, log) if cuda else 1
    prefetch = Prefetch(batcher)
    val_batches = fixed_val_batches(val_set, cfg)
    copier = Copier()

    def checkpoint():
        copier.wait()               # the previous copy reads the local file that is replaced here
        local = local_dir / "last.pt"
        save_local({"model": model.state_dict(), "logvar": logvar.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.state_dict(), "step": step, "total": total, "elapsed": elapsed,
                    "history": history, "accum": accum, "cfg": asdict(cfg), "model_cfg": config_dict(model_cfg)},
                   local)
        copier.copy(local, ck_path)
        (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")

    session_start, last_save, t_mark, s_mark = time.time(), time.time(), time.time(), step
    measure_from = step + cfg.measure_skip if total == 0 else None    # timed: measure_from .. + measure_steps
    measure_t0, skipped = None, 0
    sums, count = {}, 0
    calls = 0
    model.train()
    while total == 0 or step < total:
        if stop_after is not None and calls >= stop_after:
            break
        if cfg.stop_at and step >= cfg.stop_at:
            log(f"stopped at step {step} (stop_at)")
            break
        if (time.time() - session_start) / 3600 > cfg.session_hours:
            log("session time used up: saving; run the training cell again in a new session")
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, total, cfg)
        t_step = time.time()
        b = to_device(prefetch.get(), device)
        opt.zero_grad(set_to_none=True)
        terms = None
        try:
            terms = run_step(model, logvar, edm, b, cfg, accum, amp)
        except torch.cuda.OutOfMemoryError:
            pass
        if terms is None:                        # a rare batch did not fit: smaller parts from now on
            del b
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            accum *= 2
            log(f"out of memory: {accum} micro-batches a step from now on")
            continue
        gnorm = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(logvar.parameters()), cfg.clip))
        if not math.isfinite(gnorm):           # a non-finite gradient: skip the update
            opt.zero_grad(set_to_none=True)
            skipped += 1
            log(f"step {step}: non-finite gradient, update skipped ({skipped} in a row)")
            if skipped > 20:
                raise RuntimeError("the gradients stay non-finite: stop and look at the run")
            continue
        skipped = 0
        opt.step()
        ema.update(model)
        step += 1
        calls += 1
        elapsed += time.time() - t_step
        for k, v in {**terms, "gnorm": gnorm}.items():
            sums[k] = sums.get(k, 0.0) + v       # tensors stay on the GPU until a log line
        count += 1
        if total == 0 and step == measure_from:
            measure_t0 = time.time()
        if total == 0 and measure_t0 is not None and step == measure_from + cfg.measure_steps:
            sec = (time.time() - measure_t0) / cfg.measure_steps
            total = int(cfg.hours * 3600 / sec * 0.97)
            log(f"{sec:.3f} s a step: {total} steps in {cfg.hours} h")
        if total and any(step == int(f * total) for f in cfg.snapshots):
            ev = ema.get(0.10) if 0.10 in ema.sigma_rels else ema.models[0]
            name = f"snap_{step:07d}.pt"
            save_local({"ema": {k: v.to(torch.bfloat16) for k, v in ev.state_dict().items()}, "step": step,
                        "total": total, "model_cfg": config_dict(model_cfg), "prior": cfg.prior}, local_dir / name)
            copier.copy(local_dir / name, run_dir / name)
        if step % cfg.log_every == 0:
            rate = (step - s_mark) / max(time.time() - t_mark, 1e-6)
            row = {"step": step, "lr": lr_at(step, total, cfg), "steps_per_s": rate,
                   **{k: float(v) / count for k, v in sums.items()}}
            if total:
                row["eta_h"] = (total - step) / max(rate, 1e-6) / 3600
            history.append(row)
            log(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
            sums, count, t_mark, s_mark = {}, 0, time.time(), step
        if step % cfg.val_every == 0:
            ev = ema.get(0.10) if 0.10 in ema.sigma_rels else ema.models[0]
            v_raw = val_loss(model, edm, val_batches, cfg, device, amp)
            v_ema = val_loss(ev, edm, val_batches, cfg, device, amp)
            model.train()
            history.append({"step": step, "val_wmse": v_raw, "val_wmse_ema": v_ema})
            log(f"step {step}: validation weighted mse {v_raw:.4f} (EMA {v_ema:.4f})")
        if (time.time() - last_save) / 60 > cfg.save_minutes:
            checkpoint()
            if sheet is not None:
                try:
                    sheet(ema.get(0.10) if 0.10 in ema.sigma_rels else ema.models[0], step,
                          run_dir / "samples" / f"step_{step:07d}.png")
                except Exception as e:          # a sheet must never stop the training
                    log(f"sample sheet failed: {e!r}")
            last_save = time.time()
    checkpoint()
    copier.wait()
    done = total > 0 and step >= total
    if done:
        save_local({"ema": {s: m.state_dict() for s, m in zip(ema.sigma_rels, ema.models)}, "step": step,
                    "cfg": asdict(cfg), "model_cfg": config_dict(model_cfg), "history": history},
                   local_dir / "final.pt")
        copier.copy(local_dir / "final.pt", run_dir / "final.pt")
        copier.wait()
        log(f"finished: {step} steps, {elapsed / 3600:.2f} h")
    return {"step": step, "total": total, "elapsed_h": elapsed / 3600, "done": done}
