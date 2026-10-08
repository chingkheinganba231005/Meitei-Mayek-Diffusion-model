"""Picture sheets: canvases set out in rows, dark ink on white.

A sheet is a list of rows; a row is a list of groups separated by a grey bar; a group is a
list of canvases (float, ink 1, 64 rows) set side by side. The writer sheets of the paper
(references | real | generated) and of the owner's hand (real | few-shot | adapted) are such
sheets; the data preview shows each word above its printed prior.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .canvas import CANVAS_H
from .sample import paper


def _row(groups, gap=10):
    space = np.zeros((CANVAS_H, gap), np.float32)
    bar = np.full((CANVAS_H, 2), 0.35, np.float32)
    parts = []
    for g, group in enumerate(groups):
        if g:
            parts += [space, bar, space]
        for k, c in enumerate(group):
            if k:
                parts.append(space)
            parts.append(np.asarray(c, np.float32))
    return np.concatenate(parts, 1) if parts else space


def grid(rows, path=None, gap=10):
    """rows -> uint8 image (dark on white), saved as PNG if path is given."""
    lines = [_row(r, gap) for r in rows]
    W = max(x.shape[1] for x in lines)
    img = np.concatenate([np.pad(x, ((3, 3), (0, W - x.shape[1]))) for x in lines], 0)
    out = paper(img)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(path)
    return out


def preview(ds, path, writers=8, words=6):
    """The first words of the first writers of a rendered split, each above its printed prior
    (drawn paler)."""
    rows = []
    for w in range(min(writers, len(ds) // ds.per_writer)):
        group = []
        for i in range(w * ds.per_writer, w * ds.per_writer + min(words, ds.per_writer)):
            t, p = ds.target(i).astype(np.float32) / 255, ds.prior(i).astype(np.float32) / 255
            group.append(np.concatenate([t, np.full((2, t.shape[1]), 0.15, np.float32), 0.45 * p], 0))
        rows.append(group)
    # the rows are 2 x 64 + 2 high: stack them by hand
    lines = [np.concatenate([np.pad(c, ((0, 0), (0, 10))) for c in g], 1) for g in rows]
    W = max(x.shape[1] for x in lines)
    img = np.concatenate([np.pad(x, ((6, 6), (0, W - x.shape[1]))) for x in lines], 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(paper(img)).save(path)
    return path
