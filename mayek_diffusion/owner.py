"""The owner's real handwritten words as one writer: a fixed split, and shard sets that the
same WordSet and Batcher read.

The words (the word repository's real_words/: images/ and labels.tsv) are split once with
numpy.random.default_rng(2026).permutation(n): the first 70% are for adaptation, and the
first four of those are the fixed few-shot references; the rest are held out. Each word is
placed on the canvas with its letters' band measured on the ink (canvas.from_real), and its
printed prior is stretched over its ink, as for the synthetic words. The adaptation words
and the held-out words are written as two one-writer shard sets (adapt/, held_out/), in the
order of the split, so that items 0-3 of adapt/ are the references.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from mayek_words.charset import normalise, renderable

from .canvas import from_real, ink_columns, stretch, to_uint8
from .render import MAX_WIDTH, STYLE_KEYS

SEED = 2026
SHARE = 0.7          # of the words, for adaptation
REFERENCES = 4       # the first adaptation words: the fixed few-shot references


def read_words(folder):
    """A folder with images/ and labels.tsv -> (names, texts, uint8 greyscale images)."""
    folder = Path(folder)
    rows = [line.split("\t") for line in (folder / "labels.tsv").read_text(encoding="utf-8").splitlines() if line]
    names, texts = [r[0] for r in rows], [normalise(r[1]) for r in rows]
    bad = [t for t in texts if not renderable(t)]
    if bad:
        raise ValueError(f"words with characters outside the alphabet: {bad}")
    grays = [np.asarray(Image.open(folder / "images" / n).convert("L")) for n in names]
    return names, texts, grays


def split(n, seed=SEED, share=SHARE, refs=REFERENCES):
    """The fixed split of n words -> dict of item indices (adapt, references, held_out)."""
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(share * n))
    return {"seed": seed, "n": n, "share": share, "adapt": perm[:k].tolist(), "references": perm[:refs].tolist(),
            "held_out": perm[k:].tolist()}


def write_writer(canvases, texts, folder, printer):
    """Canvases (float, ink 1) of one writer and their texts -> a one-writer shard set in folder
    (as render.render_split writes it) -> its info."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cols_t, cols_p, offsets, ink, pw = [], [], [0], [], []
    for x, t in zip(canvases, texts):
        if x.shape[1] > MAX_WIDTH:
            raise ValueError(f"{t}: {x.shape[1]} px wide on the canvas, more than {MAX_WIDTH}")
        c, p = ink_columns(x), printer(t)
        cols_t.append(to_uint8(x))
        cols_p.append(to_uint8(stretch(p, c[0], c[1], x.shape[1])))
        offsets.append(offsets[-1] + x.shape[1])
        ink.append(c)
        pc = ink_columns(p)
        pw.append(pc[1] - pc[0])
    n = len(texts)
    np.savez_compressed(folder / "shard_0000.npz", targets=np.concatenate(cols_t, 1),
                        priors=np.concatenate(cols_p, 1), offsets=np.asarray(offsets, np.int64),
                        texts=np.asarray(texts), writer=np.zeros(n, np.int32), ink=np.asarray(ink, np.int16),
                        print_width=np.asarray(pw, np.int16),
                        style=np.full((n, len(STYLE_KEYS)), np.nan, np.float32))
    info = {"split": folder.name, "writers": 1, "words_per_writer": n, "shard_writers": 1, "real": True,
            "style_keys": STYLE_KEYS, "shards": ["shard_0000.npz"]}
    (folder / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    return info


def prepare(real_dir, out_dir, printer, seed=SEED):
    """The real words -> out_dir/adapt/ and out_dir/held_out/ (shard sets) and the split ->
    the split, with the file names and texts of its items and the ink each word lost."""
    names, texts, grays = read_words(real_dir)
    s = split(len(names), seed)
    placed = [from_real(g) for g in grays]
    for part in ("adapt", "held_out"):
        idx = s[part]
        write_writer([placed[i][0] for i in idx], [texts[i] for i in idx], Path(out_dir) / part, printer)
    s["files"] = {part: [names[i] for i in s[part]] for part in ("adapt", "references", "held_out")}
    s["texts"] = {part: [texts[i] for i in s[part]] for part in ("adapt", "references", "held_out")}
    s["ink_lost_max"] = round(max(lost for _, lost in placed), 4)
    s["widths"] = [int(x.shape[1]) for x, _ in placed]
    return s


def references(owner_dir, k=REFERENCES):
    """The fixed few-shot references (float canvases) of a prepared owner folder."""
    from .data import WordSet

    ds = WordSet(Path(owner_dir) / "adapt")
    return [ds.target(i).astype(np.float32) / 255 for i in range(k)]
