"""Pseudo-writers: the word synthesiser of the recognition project (mayek_words, pinned
commit) with a writer's habits held fixed across that writer's words.

TUMMHCD has no writer information, so a writer is made: one word style (letter height and
width, spacing, size of the signs, slant, pen, how often letters join, uneven spacing, blur,
and whether the signs above, ꯥ ꯩ ꯪ, are written beside their letter at the top), varied a
little from word to word, and for every character a small pool of images (`allographs`)
chosen among those closest in style to the writer's style anchor, so that the writer draws a
letter the same way, give or take, in every word. Words are drawn as clean ink (no paper
texture, noise or grey level), which the canvas keeps.
"""

import numpy as np

from mayek_words.charset import DIGITS, LETTERS, LONSUM
from mayek_words.glyphs import GlyphStore
from mayek_words.synth import Config, WordSynth

from .canvas import L_PX, ink_image, place

ON_LINE = LETTERS | LONSUM | DIGITS
CLEAN = dict(noise=0.0, paper=(255.0, 255.0), ink=(0.0, 0.0), min_contrast=None, blur=(0.0, 0.5))
PRINT = dict(CLEAN, letter_height_jitter=0.0, width=(1.0, 1.0), glyph_width_jitter=0.0, glyph_height_jitter=0.0,
             gap=(0.0, 0.0), gap_jitter=0.0, uneven=(0.0, 0.0), p_uneven=0.0, touch=None, mark_scale=(1.0, 1.0),
             mark_size_jitter=0.0, mark_jitter=0.0, baseline_jitter=0.0, slant=0.0, rotation=0.0,
             pen=(0.09, 0.09), style_k=None, blur=(0.0, 0.0), margin=(0.0, 0.0), p_marks_beside=0.0)


class WriterStore:
    """A GlyphStore as one writer uses it: a fixed style anchor, and per character a fixed pool
    of `allographs` images among the k closest in style to it. Without a writer
    (set_writer not called) it behaves as the store itself."""

    def __init__(self, store, allographs=2):
        self.store, self.allographs = store, allographs
        self.fixed, self.pools = None, {}

    def set_writer(self, rng):
        self.fixed, self.pools = self.store.anchor(rng), {}

    def anchor(self, rng):
        return self.fixed if self.fixed is not None else self.store.anchor(rng)

    def pick(self, char, rng, anchor=None, k=16):
        if self.fixed is None:
            return self.store.pick(char, rng, anchor, k)
        if char not in self.pools:
            idx = self.store.candidates(char)
            if k is not None and len(idx) > k:
                d = ((self.store.style[idx] - self.fixed) ** 2).sum(1)
                idx = idx[np.argpartition(d, k - 1)[:k]]
            self.pools[char] = rng.choice(idx, size=min(self.allographs, len(idx)), replace=False)
        pool = self.pools[char]
        return int(pool[rng.integers(len(pool))])

    def __getattr__(self, name):          # alpha, raw_style, candidates, ... from the store
        return getattr(self.store, name)


class WriterSynth(WordSynth):
    """WordSynth with a writer: new_writer(rng) fixes the word style and the character pools;
    every word then varies the style a little (JITTER, SHIFT) and draws its own rotation."""

    JITTER = {"L": 0.04, "width": 0.03, "mark_scale": 0.03, "pen": 0.05}    # sd of the log, per word
    SHIFT = {"slant": 0.02, "gap": 0.01}                                     # sd, per word

    def __init__(self, store, priors=None, config=None, allographs=2):
        super().__init__(WriterStore(store, allographs), priors, config)
        self.writer = None

    def new_writer(self, rng):
        self.writer = None
        self.writer = dict(super().word_style(rng))
        self.store.set_writer(rng)
        return dict(self.writer)

    def word_style(self, rng):
        if self.writer is None:
            return super().word_style(rng)
        st, c = dict(self.writer), self.cfg
        for k, sd in self.JITTER.items():
            if st[k] is not None:
                st[k] *= float(np.exp(rng.normal(0, sd)))
        for k, sd in self.SHIFT.items():
            st[k] += float(rng.normal(0, sd))
        st["rotation"] = float(np.clip(rng.normal(0, c.rotation), -2.5 * c.rotation, 2.5 * c.rotation))
        return st


def baseline_row(sample):
    """Image row of a rendered word's baseline: the median bottom of its characters that stand
    on the line (letters, lonsum letters, digits)."""
    rows = [y1 for ch, x0, y0, x1, y1 in sample.boxes if ch in ON_LINE]
    return float(np.median(rows)) if rows else float(sample.image.shape[0])


def from_sample(sample):
    """A rendered word (mayek_words.synth.Sample) -> (float32 canvas, share of ink cut off)."""
    return place(ink_image(sample.image), L_PX / sample.style["L"], baseline_row(sample))


def writer_config(**overrides):
    """The synthesiser's settings for the generator's training words: the recogniser's latest
    training words (signs above beside their letter in half of the writers), drawn clean."""
    return Config(**{**CLEAN, "p_marks_beside": 0.5, **overrides})


class Printer:
    """The printed form of a word on the canvas, the generator's content prior: the
    synthesiser's layout with the font's characters, no jitter, a pen 0.09 L wide."""

    def __init__(self, priors):
        self.synth = WordSynth(GlyphStore.from_font(), priors, Config(**PRINT))
        self.cache = {}

    def __call__(self, text):
        if text not in self.cache:
            if len(self.cache) > 50000:
                self.cache.clear()
            self.cache[text] = from_sample(self.synth.render(text, np.random.default_rng(0)))[0]
        return self.cache[text]
