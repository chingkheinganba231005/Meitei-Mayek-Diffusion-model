"""The canvas every word image is drawn on: CANVAS_H rows, ink 1 on paper 0, the baseline at
row BASE_ROW and letters L_PX pixels high, any width.

Synthetic words are placed with their known baseline and letter height (the synthesiser's
character boxes and word style). Real words are placed with the letters' band measured on
the ink (``band``): on synthetic words it finds L within about 10% (10th to 90th percentile)
and the baseline within 0.07 L, with no bias. With L_PX 22 and BASE_ROW 46 the canvas holds
2.09 L above the baseline and 0.82 L below it: 0.3% of synthetic words lose more than 0.5%
of their ink, and none of the 100 real words does.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

CANVAS_H = 64       # rows
L_PX = 22           # letter height in pixels
BASE_ROW = 46       # row of the baseline
LEFT = 5            # paper before the first ink when a word is laid out for generation, px
MULTIPLE = 32       # batched widths are multiples of this


def ink_image(gray):
    """uint8 greyscale, dark ink on light paper -> float32 in [0, 1], ink high (paper level
    the 90th percentile, ink level the 1st, as mayek_htr.images.ink_image)."""
    g = np.asarray(gray, np.float32)
    paper, ink = np.percentile(g, 90), np.percentile(g, 1)
    return np.clip((paper - g) / max(paper - ink, 16.0), 0, 1).astype(np.float32)


def resize(x, w, h):
    return np.asarray(Image.fromarray(np.ascontiguousarray(x, np.float32), "F").resize((w, h), Image.BILINEAR),
                      np.float32)


def place(ink, scale, baseline, height=CANVAS_H, base_row=BASE_ROW):
    """Ink image (float, ink high), its scale to the canvas and its baseline row -> (float32
    canvas `height` x w, the share of the ink cut off at the top or bottom)."""
    h, w = ink.shape
    nh, nw = max(int(round(h * scale)), 1), max(int(round(w * scale)), 1)
    y = resize(ink, nw, nh)
    top = int(round(base_row - baseline * scale))         # canvas row of the scaled image's first row
    out = np.zeros((height, nw), np.float32)
    src, dst = max(-top, 0), max(top, 0)
    n = min(nh - src, height - dst)
    if n > 0:
        out[dst:dst + n] = y[src:src + n]
    return np.clip(out, 0, 1), float(1.0 - out.sum() / max(float(y.sum()), 1e-6))


def to_uint8(x):
    return np.clip(np.asarray(x, np.float32) * 255 + 0.5, 0, 255).astype(np.uint8)


def ink_columns(x, threshold=0.1):
    """(first, last + 1) column holding ink, or None."""
    cols = np.flatnonzero(np.asarray(x).max(0) > threshold * (255 if np.asarray(x).dtype == np.uint8 else 1))
    return (int(cols[0]), int(cols[-1]) + 1) if len(cols) else None


def stretch(prior, x0, x1, width):
    """The prior's ink resampled to span columns x0..x1 of a canvas `width` wide (float32)."""
    c = ink_columns(prior)
    out = np.zeros((prior.shape[0], width), np.float32)
    if c is None or x1 <= x0:
        return out
    body = resize(np.asarray(prior, np.float32)[:, c[0]:c[1]], x1 - x0, prior.shape[0])
    out[:, x0:min(x1, width)] = body[:, :min(x1, width) - x0]
    return out


# ------------------------------------------------------------------ real words

def pieces(gray, level, min_px=20):
    """Boxes (y0, y1, x0, x1) of the 8-connected pieces darker than `level`, of min_px or more."""
    lab, _ = ndimage.label(np.asarray(gray) < level, structure=np.ones((3, 3)))
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is not None and int((lab[sl] == k).sum()) >= min_px:
            out.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return out


def band(gray):
    """(top line row, baseline row, L) of a word image (dark ink on light paper): the medians
    of the tops and of the bottoms of its large pieces of ink (at least 0.6 of the tallest),
    as scripts/measure_sign_geometry.py measures the real words; None without ink."""
    g = np.asarray(gray, np.float32)
    level = (np.percentile(g, 90) + np.percentile(g, 1)) / 2
    ps = pieces(gray, level)
    if not ps:
        return None
    tallest = max(p[1] - p[0] for p in ps)
    big = [p for p in ps if p[1] - p[0] >= 0.6 * tallest]
    top, base = float(np.median([p[0] for p in big])), float(np.median([p[1] for p in big]))
    return top, base, base - top


def crop_ink(gray, margin=0.15):
    """A word cut out of a larger image: the ink's box plus margin x its height on every side
    (mayek_htr.images.crop_ink)."""
    x = ink_image(gray)
    ys, xs = np.nonzero(x > 0.5)
    if not len(xs):
        return np.asarray(gray)
    pad = int(margin * (ys.max() - ys.min() + 1)) + 1
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, x.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, x.shape[1])
    return np.asarray(gray)[y0:y1, x0:x1]


def from_real(gray, crop=True):
    """A real word image (uint8 greyscale, dark on light; one word) -> (float32 canvas, share of
    ink cut off). Small images are enlarged first so that the band can be measured."""
    g = crop_ink(gray) if crop else np.asarray(gray)
    if g.shape[0] < 96:
        f = 96 / g.shape[0]
        g = np.clip(resize(g.astype(np.float32), max(int(round(g.shape[1] * f)), 1), 96), 0, 255).astype(np.uint8)
    b = band(g)
    if b is None or b[2] < 4:
        raise ValueError("no ink found in the image")
    top, base, L = b
    return place(ink_image(g), L_PX / L, base)
