"""Writing words in a writer's hand: reference canvases and texts in, canvases out.

The width of each word comes from its printed width times the writer's ratio predicted by the
style encoder (times width_scale, a user control); its printed form is stretched over that
width, LEFT pixels from the left edge, and the canvas is rounded up to a multiple of 32.
"""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .canvas import CANVAS_H, LEFT, MULTIPLE, ink_columns, stretch
from .data import encode
from .edm import EDM, dpmpp_2m, guided, heun, karras_sigmas

SAMPLERS = {"heun": heun, "dpmpp": dpmpp_2m}


def refs_tensor(groups, device, max_width=256):
    """groups: per word, a list of float canvases (64, w), ink 1 -> ref (B, K, 64, R) in [-1, 1],
    ref_width (B, K), ref_mask (B, K)."""
    K = max(len(g) for g in groups)
    R = min(max(-(-c.shape[1] // 16) * 16 for g in groups for c in g), max_width)
    ref = -np.ones((len(groups), K, CANVAS_H, R), np.float32)
    rw = np.zeros((len(groups), K), np.int64)
    rm = np.zeros((len(groups), K), bool)
    for b, g in enumerate(groups):
        for k, c in enumerate(g):
            c = np.asarray(c, np.float32)[:, :R]
            ref[b, k, :, :c.shape[1]] = 2 * c - 1
            rw[b, k], rm[b, k] = c.shape[1], True
    return torch.from_numpy(ref).to(device), torch.from_numpy(rw).to(device), torch.from_numpy(rm).to(device)


def text_tensor(texts, device):
    ids = [encode(t) for t in texts]
    out = np.zeros((len(ids), max(len(x) for x in ids)), np.int64)
    for b, x in enumerate(ids):
        out[b, :len(x)] = x
    return torch.from_numpy(out).to(device)


def layout(printed, log_ratio, width_scale=1.0):
    """Printed canvas, log(handwriting width / printed width) -> (prior canvas, width of the ink)."""
    c = ink_columns(printed)
    w = max(int(round((c[1] - c[0]) * math.exp(log_ratio) * width_scale)), 4)
    W = -(-(2 * LEFT + w) // MULTIPLE) * MULTIPLE
    return stretch(printed, LEFT, LEFT + w, W), w


@torch.no_grad()
def generate(model, texts, refs, printer, edm=None, steps=16, sampler="dpmpp", guidance=2.0,
             interval=(0.3, 5.0), width_scale=1.0, seed=0, device="cpu", dtype=None, prior=True):
    """texts: words; refs: the reference canvases of one writer (a list of float (64, w)
    arrays), or one such list per word, or None for a random hand (the null style, without
    guidance, at the printed width) -> list of float32 canvases (64, w), ink 1, each cut LEFT
    px after its ink.

    Words are denoised in groups of the same canvas width, so that no word has more paper to
    its right than in training (less than 32 px). prior=False leaves the printed prior blank,
    for a model trained without it (the ablation)."""
    edm, prior_on = edm or EDM(), prior
    model.eval()
    B = len(texts)
    random_hand = refs is None
    if random_hand:
        refs, guidance = [np.zeros((CANVAS_H, 16), np.float32)], 1.0
    groups = [refs] * B if isinstance(refs[0], np.ndarray) else list(refs)
    dev = torch.device(device)
    amp = torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None)
    ref, rw, rm = refs_tensor(groups, dev)
    with amp:
        g_ref = model.style(ref, rw, rm)[2]
        log_ratio = model.style.width(g_ref)[:, 0].float().cpu().numpy()
    if random_hand:
        log_ratio = np.zeros(B)
    priors, widths = zip(*(layout(printer(t), r, width_scale) for t, r in zip(texts, log_ratio)))
    gen = torch.Generator().manual_seed(seed)
    noise = [torch.randn(1, 1, CANVAS_H, p.shape[1], generator=gen) for p in priors]
    sig = karras_sigmas(steps)
    out = [None] * B
    for W in sorted({p.shape[1] for p in priors}):
        idx = [b for b in range(B) if priors[b].shape[1] == W]
        ids = text_tensor([texts[b] for b in idx], dev)
        r, w_, m = ref[idx], rw[idx], rm[idx]
        null = torch.ones(len(idx), dtype=torch.bool, device=dev)
        with amp:
            uncond = model.condition(ids, r, w_, m, drop_style=null)
            cond = uncond if random_hand else model.condition(ids, r, w_, m)
        prior = torch.from_numpy(np.stack([2 * priors[b] - 1 for b in idx])[:, None]).to(dev)
        cond["prior"] = uncond["prior"] = prior if prior_on else torch.full_like(prior, -1.0)

        def denoise(x, sigma, c):
            net = lambda x_in, c_noise: model(x_in, c["prior"], c_noise, c)
            with amp:
                return edm.denoise(net, x, torch.full((x.shape[0],), sigma, device=dev))

        x = torch.cat([noise[b] for b in idx]).to(dev) * sig[0]
        x = SAMPLERS[sampler](guided(denoise, cond, uncond, guidance, interval), x, sig)
        ink = ((x.float() + 1) / 2).clamp(0, 1).cpu().numpy()[:, 0]
        for j, b in enumerate(idx):
            out[b] = ink[j, :, :min(W, 2 * LEFT + widths[b])]
    return out


def paper(ink):
    """Float canvas, ink 1 -> uint8 dark ink on white."""
    return 255 - np.clip(np.asarray(ink, np.float32) * 255 + 0.5, 0, 255).astype(np.uint8)


def sheet(model, val_set, printer, path, writers=6, refs=4, steps=16, device="cpu", dtype=None, prior=True):
    """A progress sheet: per validation writer, its references, then two of its other words as
    written (left) and as generated (right)."""
    rows, per = [], val_set.per_writer
    for w in range(min(writers, len(val_set) // per)):
        items = list(range(w * per, (w + 1) * per))
        ref = [val_set.target(i).astype(np.float32) / 255 for i in items[:refs]]
        real = [val_set.target(i).astype(np.float32) / 255 for i in items[refs:refs + 2]]
        gen = generate(model, [val_set.texts[i] for i in items[refs:refs + 2]], ref, printer, steps=steps,
                       seed=w, device=device, dtype=dtype, prior=prior)
        gap = np.zeros((CANVAS_H, 12), np.float32)
        bar = np.full((CANVAS_H, 2), 0.5, np.float32)
        parts = [p for r in ref for p in (r, gap)] + [bar, gap]
        for a, b in zip(real, gen):
            parts += [a, gap, b, gap, bar, gap]
        rows.append(np.concatenate(parts, 1))
    W = max(r.shape[1] for r in rows)
    img = np.concatenate([np.pad(r, ((2, 2), (0, W - r.shape[1]))) for r in rows], 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(paper(img)).save(path)
