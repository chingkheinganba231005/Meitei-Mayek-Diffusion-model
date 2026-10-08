"""Generated words as training data for the word recogniser (E6).

The generator writes training-lexicon words in one hand (the owner's, from the four fixed
references); the deployed recogniser is then trained further for a few thousand steps on the
synthesiser's words alone, or on half synthesiser and half generated words, with its own
augmentation and CTC loss, and the copies are compared on real words. The word repository's
training script cannot start from a word checkpoint, hence this small loop. The recogniser
code (mayek_htr) needs timm.
"""

import io
import math
import tarfile
import time
from pathlib import Path

import numpy as np

from .evaluate import word_png
from .sample import generate, layout

MAX_PRINTED = 300        # texts printed wider than this (px on the canvas) are left out: the generator saw <= 384


def draw_texts(lexicon, n, seed=300, scrambled=0.1, printer=None):
    """n training texts as the synthesiser draws them (mayek_words.synth.Words.text; item i from
    the seed sequence (seed, i)), leaving out those printed wider than MAX_PRINTED."""
    from mayek_words.synth import Words

    from .canvas import ink_columns

    words, out, i = Words(None, lexicon, seed=seed, scrambled=scrambled), [], 0
    while len(out) < n:
        t = words.text(np.random.default_rng([seed, i]))
        i += 1
        if printer is not None:
            c = ink_columns(printer(t))
            if c is None or c[1] - c[0] > MAX_PRINTED:
                continue
        out.append(t)
    return out


def write_generated(model, texts, refs, printer, path, batch=256, seed=0, log=print, **kw):
    """Generate the texts in one hand (refs) and write them as a fixed set (.tar: images/,
    labels.tsv), cut to their ink as the real words were. Batches hold words of similar
    printed width."""
    order = sorted(range(len(texts)), key=lambda i: layout(printer(texts[i]), 0.0)[1])
    rows, t0 = [None] * len(texts), time.time()
    tmp = Path(str(path) + ".tmp")
    with tarfile.open(tmp, "w") as tar:
        for k in range(0, len(order), batch):
            chunk = order[k:k + batch]
            for i, g in zip(chunk, generate(model, [texts[i] for i in chunk], refs, printer, seed=seed + k, **kw)):
                data, name = word_png(g), f"{i + 1:06d}.png"
                info = tarfile.TarInfo("images/" + name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                rows[i] = f"{name}\t{texts[i]}"
            log(f"{min(k + batch, len(order))} of {len(order)} words, {time.time() - t0:.0f} s")
        data = ("\n".join(rows) + "\n").encode("utf-8")
        info = tarfile.TarInfo("labels.tsv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    tmp.replace(path)
    return len(texts)


def real_set(real_dir, exclude, path):
    """The real words except `exclude` (file names), as a fixed set .tar of the original images."""
    real_dir = Path(real_dir)
    rows = [line.split("\t") for line in (real_dir / "labels.tsv").read_text(encoding="utf-8").splitlines() if line]
    kept = [r for r in rows if r[0] not in set(exclude)]
    with tarfile.open(path, "w") as tar:
        for name, _ in kept:
            tar.add(real_dir / "images" / name, arcname="images/" + name)
        data = ("".join(f"{n}\t{t}\n" for n, t in kept)).encode("utf-8")
        info = tarfile.TarInfo("labels.tsv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return len(kept)


def _items(batch):
    """A batch of mayek_htr.data (x uint8 (B, 1, H, W), widths, texts) -> [(image, text)]."""
    x, w = batch["x"].numpy(), batch["widths"].numpy()
    return [(x[i, 0, :, :w[i]], t) for i, t in enumerate(batch["texts"])]


def finetune_recogniser(checkpoint, words, out, generated=None, mix=0.5, steps=5000, batch=64, lr=1e-4,
                        warmup=200, workers=8, device=None, seed=0, log=print):
    """Train the recogniser of `checkpoint` (recogniser.pt or best.pt) further on synthetic words
    (mayek_words.synth.Words), and on generated words (a fixed set) in share `mix` of each batch
    if given; learning rate warmed up over `warmup` steps, then constant; an exponential average
    of the weights (the recogniser's own decay) is saved to `out` as eval_recogniser.py reads it."""
    import torch
    import torch.nn as nn

    from mayek_htr.augment import PRESETS, augment
    from mayek_htr.data import FixedSet, SynthStream, make_batch
    from mayek_htr.model import MEAN, STD
    from mayek_htr.train import EMA, TrainConfig, amp_dtype, load_checkpoint, param_groups, prepare, save

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model, ck = load_checkpoint(checkpoint, device)
    cfg = TrainConfig(**ck["cfg"])
    model.train()
    ema = EMA(model, cfg.ema)
    opt = torch.optim.AdamW(param_groups(model, cfg.wd), lr=lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    amp = amp_dtype(device)
    n_gen = int(round(batch * mix)) if generated is not None else 0
    gen = FixedSet(generated) if n_gen else None
    rng = np.random.default_rng(seed)
    stream = SynthStream(words, batch - n_gen, cfg.pool)
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=workers,
                                         persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    batches, history, recent, t0 = iter(loader), [], [], time.time()
    for step in range(steps):
        items = _items(next(batches))
        if gen is not None:
            items += [(gen.images[i], gen.texts[i]) for i in rng.choice(len(gen), n_gen, replace=False)]
        b = make_batch(items)
        x, widths = prepare(b, device)
        x = (augment(x, widths, PRESETS[cfg.aug]) - MEAN) / STD
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            logits, lengths = model(x, widths)
        logp = logits.float().log_softmax(-1).transpose(0, 1)
        loss = ctc(logp, b["targets"].to(device), lengths, b["target_lengths"].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        ema.update(model)
        recent.append(loss.item())
        if not math.isfinite(recent[-1]):
            raise RuntimeError("the loss is not finite")
        if (step + 1) % 200 == 0 or step + 1 == steps:
            row = {"step": step + 1, "loss": round(float(np.mean(recent)), 5),
                   "words_per_s": round((step + 1) * batch / (time.time() - t0), 1)}
            history.append(row)
            log(f"step {row['step']}  loss {row['loss']:.4f}  {row['words_per_s']:.0f} words/s")
            recent = []
    save({"model": ema.module.state_dict(), "cfg": ck["cfg"], "step": steps,
          "finetune": {"from": str(checkpoint), "steps": steps, "batch": batch, "lr": lr, "warmup": warmup,
                       "generated": str(generated) if generated else None, "mix": mix if n_gen else 0.0,
                       "history": history}}, out)
    return history
