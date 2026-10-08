"""Rendered splits in memory, and training batches.

A batch has one width W (a multiple of 32): its words are those whose canvas fits W and not
W - 32, so little is padding, and the number of words is set by a pixel budget (128 words at
W = 256). Each word comes with 1 to `refs` other words of its writer (its style references,
right-padded to a common width, cut at max_ref_width), its text as character ids, and its
printed prior. Batches hold at most max_batch words (the narrowest words would otherwise come
512 at a time, with 4 references each).
"""

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mayek_words.charset import ALPHABET

from .canvas import CANVAS_H, MULTIPLE

CHAR_ID = {ch: i + 1 for i, ch in enumerate(ALPHABET)}     # 0 is padding
NUM_CHARS = len(ALPHABET)                                  # 54; ids 1..54, NULL_ID for "no text"
NULL_ID = NUM_CHARS + 1


def encode(text):
    return [CHAR_ID[ch] for ch in text]


class WordSet:
    """The shards of a split (render.render_split), kept in memory as uint8."""

    def __init__(self, folder, limit_shards=None):
        folder = Path(folder)
        self.info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
        names = self.info["shards"][:limit_shards] if limit_shards else self.info["shards"]
        with ThreadPoolExecutor(8) as pool:                 # decompression runs outside the GIL
            self.shards = list(pool.map(lambda n: dict(np.load(folder / n)), names))
        self.per_writer = self.info["words_per_writer"]
        counts = [len(s["texts"]) for s in self.shards]
        self.start = np.concatenate([[0], np.cumsum(counts)])
        self.texts = np.concatenate([s["texts"] for s in self.shards])
        self.writer = np.concatenate([s["writer"] for s in self.shards])
        self.ink = np.concatenate([s["ink"] for s in self.shards]).astype(np.int64)
        self.print_width = np.concatenate([s["print_width"] for s in self.shards]).astype(np.int64)
        self.widths = np.concatenate([np.diff(s["offsets"]) for s in self.shards]).astype(np.int64)
        assert len(self.texts) % self.per_writer == 0
        assert (self.writer[::self.per_writer] == self.writer[self.per_writer - 1::self.per_writer]).all()

    def __len__(self):
        return len(self.texts)

    def _loc(self, i):
        k = int(np.searchsorted(self.start, i, side="right") - 1)
        j = int(i - self.start[k])
        s = self.shards[k]
        return s, int(s["offsets"][j]), int(s["offsets"][j + 1])

    def target(self, i):
        s, a, b = self._loc(i)
        return s["targets"][:, a:b]

    def prior(self, i):
        s, a, b = self._loc(i)
        return s["priors"][:, a:b]

    def writer_items(self, i):
        """The items of item i's writer."""
        first = (i // self.per_writer) * self.per_writer
        return np.arange(first, first + self.per_writer)


class Batcher:
    def __init__(self, ds, pixels=128 * 256, refs=4, max_ref_width=256, max_width=384, min_batch=8, max_batch=256,
                 seed=0, p_all_refs=0.5):
        self.ds, self.pixels, self.refs, self.max_ref_width = ds, pixels, refs, max_ref_width
        self.min_batch, self.max_batch, self.p_all_refs = min_batch, max_batch, p_all_refs
        self.rng = np.random.default_rng(seed)
        bucket = -(-ds.widths // MULTIPLE) * MULTIPLE
        self.buckets = {int(W): np.flatnonzero(bucket == W) for W in np.unique(bucket) if W <= max_width}
        self.Ws = np.array(sorted(self.buckets))
        # a bucket is drawn in proportion to its words over its batch size, so that every word
        # is drawn about equally often
        n = np.array([len(self.buckets[W]) / self.batch_size(W) for W in self.Ws], np.float64)
        self.p = n / n.sum()

    def batch_size(self, W):
        return min(self.max_batch, max(self.min_batch, (self.pixels // W) // 8 * 8))

    def sample(self, W=None):
        """-> dict of numpy arrays for one batch (of width W, or of a bucket drawn at random)."""
        rng, ds = self.rng, self.ds
        W = int(rng.choice(self.Ws, p=self.p)) if W is None else int(W)
        pool = self.buckets[W]
        B = min(self.batch_size(W), len(pool))
        idx = rng.choice(pool, B, replace=False)
        target = np.zeros((B, CANVAS_H, W), np.uint8)
        prior = np.zeros((B, CANVAS_H, W), np.uint8)
        refs, n_refs = [], np.zeros(B, np.int64)
        for b, i in enumerate(idx):
            t, p = ds.target(i), ds.prior(i)
            target[b, :, :t.shape[1]] = t
            prior[b, :, :p.shape[1]] = p
            others = ds.writer_items(i)
            others = others[others != i]
            k = self.refs if rng.random() < self.p_all_refs else int(rng.integers(1, self.refs + 1))
            chosen = rng.choice(others, k, replace=False)
            refs.append(chosen)
            n_refs[b] = k
        rw = max(int(ds.widths[j]) for r in refs for j in r)
        R = min(-(-rw // 16) * 16, self.max_ref_width)
        ref = np.zeros((B, self.refs, CANVAS_H, R), np.uint8)
        ref_w = np.zeros((B, self.refs), np.int64)
        for b, r in enumerate(refs):
            for s, j in enumerate(r):
                im = ds.target(j)[:, :R]
                ref[b, s, :, :im.shape[1]] = im
                ref_w[b, s] = im.shape[1]
        ids = [encode(ds.texts[i]) for i in idx]
        T = max(len(x) for x in ids)
        text = np.zeros((B, T), np.int64)
        for b, x in enumerate(ids):
            text[b, :len(x)] = x
        ink = ds.ink[idx]
        return {"target": target, "prior": prior, "ref": ref, "ref_width": ref_w,
                "ref_mask": np.arange(self.refs)[None] < n_refs[:, None], "text": text,
                "writer": ds.writer[idx].astype(np.int64), "width": ds.widths[idx],
                "log_width_ratio": np.log((ink[:, 1] - ink[:, 0]) / ds.print_width[idx]).astype(np.float32),
                "index": idx}


class Prefetch:
    """Batches made on a background thread (numpy copies release the GIL)."""

    def __init__(self, batcher, depth=6):
        self.q = queue.Queue(depth)
        self.batcher = batcher
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            while True:
                self.q.put(self.batcher.sample())
        except Exception as e:          # surfaced on the next get
            self.error = e
            self.q.put(None)

    def get(self):
        b = self.q.get()
        if b is None:
            raise RuntimeError("batch thread failed") from self.error
        return b
