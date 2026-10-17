"""Reference consistency: a writer's hand should not depend on which of their words are shown.

A few reference words show only some of a writer's letters, and the generator may copy
incidental details of the words it happens to be given. For a target word, the training draws
two disjoint sets of other words of its writer that show the same letters of the target: the
characters of the target found in set A are exactly those found in set B. With the same noisy
target, noise level and printed prior, the denoiser runs once with each set, and a penalty asks
the two predictions to agree, weighted by the share of the target's characters the sets show.
Where two sets show different letters of the target, their predictions may rightly differ, so
unmatched pairs carry no penalty.

What the penalty does. For two sets with predictions a and b of a pixel whose best estimates are
m_A and m_B, minimising (a - y)^2 + (b - y)^2 + lam (a - b)^2 gives
a = (1 + lam) / (1 + 2 lam) m_A + lam / (1 + 2 lam) m_B: each prediction moves a share
lam / (1 + 2 lam) of the way towards what another, equally informative set of the same writer
would give (a quarter of the way for lam = 0.5). Matching the sets on the target's letters keeps
the letter-specific evidence in both, so what is averaged away is the choice of reference words.
The difference D_A - D_B = c_out (F_A - F_B) at the same x and sigma is divided by c_out^2, so
the penalty compares the network outputs, on the scale of EDM's weighted loss.

Sampling (PairedBatcher). Set A: 1 to 4 other words of the writer (all 4 half the time), never a
word with the target's text, one word per distinct text, at most 384 px wide (whole words: a
cut word would show fewer letters than its text says). Set B: as many further words with the
same coverage of the target's characters, searched among all the writer's remaining words; if
none exists, A is drawn again (4 tries), then the sets are made smaller; otherwise the pair is
left unmatched. The transcripts are used by the training only: the generator sees images.

The variants of the training study (TrainConfig.reference_objective) all draw set A the same way:
  single    set A only: one denoising pass (the baseline)
  paired    a second pass with set B on a share of the batch, supervised, no penalty
  ungated   the penalty with weight 1 on a random set B of the same size (unmatched)
  random    matched pairs, the coverage weights shuffled within the batch (same mass, wrong pairs)
  coverage  matched pairs, the penalty weighted by coverage (the proposed objective)
"""

import itertools

import numpy as np
import torch

from .canvas import CANVAS_H
from .data import Batcher

OBJECTIVES = ("baseline", "single", "paired", "ungated", "random", "coverage")
TWO_PASSES = ("paired", "ungated", "random", "coverage")
_COMBOS = {}


def combos(n, k):
    """All k-subsets of range(n) as an int array (cached)."""
    if (n, k) not in _COMBOS:
        _COMBOS[(n, k)] = np.array(list(itertools.combinations(range(n), k)), np.int64).reshape(-1, k)
    return _COMBOS[(n, k)]


def coverage(target, texts):
    """The characters of `target` that appear in `texts`."""
    return set(target) & set("".join(texts))


def match_sets(target, texts, k, rng, tries=4, matched=True, pool=15):
    """Two disjoint sets of positions in `texts` (candidate references of one writer, distinct
    texts) -> (a, b, weight): b covers the same characters of `target` as a, and weight is the
    share of the target's characters they cover; (a, b, 0.0) for an unmatched pair; None if fewer
    than two candidates. matched=False: a is drawn as before, b is a random disjoint set of the
    same size (the ungated variant), and the weight is 1. A writer with more than `pool` other
    words (a real writer) is searched among `pool` of them, drawn at random, as in training."""
    if len(texts) > pool:
        keep = sorted(rng.choice(len(texts), pool, replace=False).tolist())
        found = match_sets(target, [texts[i] for i in keep], k, rng, tries, matched, pool)
        return found and ([keep[i] for i in found[0]], [keep[i] for i in found[1]], found[2])
    n = len(texts)
    if n < 2:
        return None
    chars = sorted(set(target))
    bit = {c: 1 << i for i, c in enumerate(chars)}
    masks = np.array([sum(bit[c] for c in set(t) if c in bit) for t in texts], np.int64)
    found = None
    for kk in range(min(k, n // 2), 0, -1):
        for _ in range(tries):
            perm = rng.permutation(n)
            a, rest = perm[:kk], perm[kk:]
            ma = int(np.bitwise_or.reduce(masks[a]))
            if not ma:
                continue
            cand = rest[combos(len(rest), kk)]
            hits = np.flatnonzero(np.bitwise_or.reduce(masks[cand], axis=1) == ma)
            if len(hits):
                found = a.tolist(), cand[int(rng.choice(hits))].tolist(), bin(ma).count("1") / len(chars)
                break
        if found:
            break
    if found is None:
        perm = rng.permutation(n)
        kk = max(min(k, n // 2), 1)
        found = perm[:kk].tolist(), perm[kk:2 * kk].tolist(), 0.0
    if not matched:
        rest = [x for x in rng.permutation(n).tolist() if x not in found[0]]
        found = found[0], rest[:len(found[0])], 1.0
    return found


class PairedBatcher(Batcher):
    """Batches as Batcher gives them, with references drawn as above: set A in `ref`, set B in
    `ref_alt`, the pair's weight in `pair_weight`, whether the pair is usable in `pair_valid`,
    and the rows that get the second pass in `second` (a share pair_fraction of the batch)."""

    def __init__(self, ds, *args, matched=True, pair_fraction=0.5, **kwargs):
        kwargs.setdefault("max_ref_width", 384)
        super().__init__(ds, *args, **kwargs)
        self.matched, self.pair_fraction = matched, pair_fraction

    def sample(self, W=None):
        b = super().sample(W)
        ds, rng = self.ds, self.rng
        pairs, valid, weights = [], [], []
        for i in b["index"]:
            by_text = {}
            for j in ds.writer_items(i):
                if ds.texts[j] != ds.texts[i] and ds.widths[j] <= self.max_ref_width:
                    by_text.setdefault(str(ds.texts[j]), []).append(int(j))
            items = [int(rng.choice(v)) for v in by_text.values()]
            k = self.refs if rng.random() < self.p_all_refs else int(rng.integers(1, self.refs + 1))
            m = match_sets(str(ds.texts[i]), list(by_text), k, rng, matched=self.matched)
            if m is None:            # fewer than two other texts: a reference, no pair
                others = items or [int(j) for j in ds.writer_items(i) if j != i]
                pairs.append(([others[0]], [others[0]]))
                valid.append(False)
                weights.append(0.0)
            else:
                pairs.append(([items[x] for x in m[0]], [items[x] for x in m[1]]))
                valid.append(True)
                weights.append(m[2])
        R = min(max(int(ds.widths[j]) for pair in pairs for side in pair for j in side), self.max_ref_width)
        R = -(-R // 16) * 16
        B = len(pairs)
        for side, suffix in ((0, ""), (1, "_alt")):
            ref = np.zeros((B, self.refs, CANVAS_H, R), np.uint8)
            width = np.zeros((B, self.refs), np.int64)
            mask = np.zeros((B, self.refs), bool)
            index = np.full((B, self.refs), -1, np.int64)
            for row, pair in enumerate(pairs):
                for slot, j in enumerate(pair[side]):
                    im = ds.target(j)[:, :R]
                    ref[row, slot, :, :im.shape[1]] = im
                    width[row, slot], mask[row, slot], index[row, slot] = im.shape[1], True, j
            b["ref" + suffix], b["ref_width" + suffix] = ref, width
            b["ref_mask" + suffix], b["ref_index" + suffix] = mask, index
        second = np.zeros(B, bool)
        second[rng.permutation(B)[:int(round(self.pair_fraction * B))]] = True
        b.update(pair_valid=np.asarray(valid, bool), pair_weight=np.asarray(weights, np.float32), second=second)
        return b


def consistency(da, db, sigma, edm, weight):
    """Per word: (D_A - D_B)^2 averaged over pixels, divided by c_out^2, times weight."""
    c_out = edm.coefficients(sigma.float())[1]
    per = (da.float() - db.float()).square().mean((1, 2, 3)) / c_out.square().clamp_min(1e-12)
    return per * weight


def pair_weights(b, objective, sigma, drop_style, interval):
    """The penalty's weight for every row of a batch (zero where it does not apply)."""
    if objective == "ungated":
        w = b["pair_valid"].float()
    elif objective == "random":
        w = b["pair_weight"][torch.randperm(len(sigma), device=sigma.device)]
    elif objective == "coverage":
        w = b["pair_weight"]
    else:
        return torch.zeros_like(sigma)
    on = b["pair_valid"] & ~drop_style & (sigma >= interval[0]) & (sigma <= interval[1])
    return w.float() * on.float()
