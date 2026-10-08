"""Helpers of the demo (space/app.py): the on-screen keyboard, the text typed, the style
references, and the words set on one line.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from mayek_words.charset import ALPHABET, DIGITS, LETTERS, LONSUM, SIGNS, normalise

from .canvas import CANVAS_H, L_PX, LEFT, from_real, ink_columns
from .sample import paper

MAX_WORDS = 6
MAX_CHARS = 20
GAP = 0.5            # between words, in letter heights


def keyboard():
    """The 54 characters of everyday spelling in groups -> [(name, characters)]."""
    groups = [("Letters", LETTERS), ("Lonsum", LONSUM), ("Signs", SIGNS), ("Digits", DIGITS)]
    out = [(name, [c for c in ALPHABET if c in chars]) for name, chars in groups]
    used = {c for _, cs in out for c in cs}
    out.append(("Full stop and apun", [c for c in ALPHABET if c not in used]))
    return out


def parse(text, max_words=MAX_WORDS, max_chars=MAX_CHARS):
    """Typed text -> words in everyday spelling. Raises ValueError with a message for the page."""
    words = normalise(text or "").split()
    if not words:
        raise ValueError("Type a word in Meitei Mayek, or use the keyboard.")
    if len(words) > max_words:
        raise ValueError(f"At most {max_words} words at a time.")
    allowed = set(ALPHABET)
    for w in words:
        bad = sorted({c for c in w if c not in allowed})
        if bad:
            raise ValueError(f"{w}: only Meitei Mayek letters, signs, digits, ꯫ and apun can be written "
                             f"(not {' '.join(bad)}).")
        if len(w) > max_chars:
            raise ValueError(f"{w}: at most {max_chars} characters a word.")
    return words


def compose_line(canvases, gap=GAP):
    """Generated canvases (sharing the baseline) -> one canvas: each word cut to its ink, the
    words gap x L apart, LEFT px of paper at both ends."""
    space = np.zeros((CANVAS_H, int(round(gap * L_PX))), np.float32)
    parts = [np.zeros((CANVAS_H, LEFT), np.float32)]
    for k, c in enumerate(canvases):
        cols = ink_columns(c)
        word = np.asarray(c, np.float32)[:, cols[0]:cols[1]] if cols else np.zeros((CANVAS_H, L_PX), np.float32)
        parts += ([space] if k else []) + [word]
    parts.append(np.zeros((CANVAS_H, LEFT), np.float32))
    return np.concatenate(parts, 1)


def enlarge(canvas, factor=3):
    """Float canvas -> uint8 image, dark on white, `factor` times larger."""
    im = Image.fromarray(paper(canvas))
    return np.asarray(im.resize((im.width * factor, im.height * factor), Image.BICUBIC))


def references_from_images(images):
    """Uploaded word images (uint8 greyscale or RGB arrays; one word each) -> (canvases, messages
    for the images that could not be used)."""
    refs, notes = [], []
    for k, im in enumerate(images, 1):
        g = np.asarray(Image.fromarray(np.asarray(im)).convert("L"))
        try:
            x, lost = from_real(g)
        except ValueError as e:
            notes.append(f"image {k}: {e}")
            continue
        if x.shape[1] > 384:
            notes.append(f"image {k}: too wide for one word; cut to its first 384 px on the canvas")
            x = x[:, :384]
        refs.append(x)
    return refs, notes


def save_canvases(canvases, folder):
    """Canvases -> folder/1.png, 2.png, ... (dark on white, 64 rows)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(canvases, 1):
        Image.fromarray(paper(c)).save(folder / f"{k}.png")


def load_presets(folder):
    """folder/<name>/<k>.png (save_canvases) -> {name: canvases}, names in order."""
    out = {}
    for d in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        files = sorted(d.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        if files:
            out[d.name] = [1 - np.asarray(Image.open(f).convert("L"), np.float32) / 255 for f in files]
    return out
