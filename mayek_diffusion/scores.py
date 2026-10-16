"""Scores of generated words, as the evaluation protocol defines them.

- Legibility: the deployed word recogniser (its recogniser.pt, char_lm.pkl and
  decoding.json from Hugging Face) reads a set of word images through the word repository's
  scripts/eval_recogniser.py: character and word error rates, greedy and with the language
  model at the weights in decoding.json.
- Style: HWD (Pippi et al., BMVC 2023) between the generated and the real words of each
  writer, averaged over the writers, and FID and KID over the whole set, with the
  implementations of the HWD package (https://github.com/aimagelab/HWD). Images are read
  from one folder per writer (evaluate.write_writer_folders).
- The rule that picks a sampling setting on validation: the lowest HWD among the settings
  whose greedy CER is at most the reference CER plus 1 percentage point; if none qualifies,
  the lowest CER.
- Bootstrap intervals of CER and WER for small sets.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

KEEP = ("words", "characters", "cer", "wer", "word_accuracy", "word_accuracy_95ci")


def recogniser_files(folder):
    """The deployed recogniser's files in a folder (as on Hugging Face) -> dict."""
    folder = Path(folder)
    decoding = json.loads((folder / "decoding.json").read_text(encoding="utf-8"))
    return {"checkpoint": folder / "recogniser.pt", "lm": folder / "char_lm.pkl", "alpha": decoding["alpha"],
            "beta": decoding["beta"], "beam": decoding.get("beam", 16)}


def recognise(fixed_set, recogniser, word_repo, out, predictions=None, lm=True, device=None, processes=None):
    """The recogniser reads a fixed set (.tar or folder) -> {words, greedy, with_lm} (CER, WER,
    word accuracy); the full results go to `out`."""
    r = recogniser_files(recogniser) if not isinstance(recogniser, dict) else recogniser
    cmd = [sys.executable, "-u", str(Path(word_repo) / "scripts" / "eval_recogniser.py"),
           "--checkpoint", str(r["checkpoint"]), "--set", str(fixed_set), "--out", str(out)]
    if lm:
        cmd += ["--lm", str(r["lm"]), "--alpha", str(r["alpha"]), "--beta", str(r["beta"]), "--beam", str(r["beam"])]
    if predictions:
        cmd += ["--predictions", str(predictions)]
    if device:
        cmd += ["--device", str(device)]
    if processes:
        cmd += ["--processes", str(processes)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"eval_recogniser.py failed:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")
    res = json.loads(Path(out).read_text(encoding="utf-8"))
    return {"words": res["words"], "greedy": {k: res["greedy"][k] for k in KEEP},
            "with_lm": {k: res["with_lm"][k] for k in KEEP} if "with_lm" in res else None}


def frechet(f1, f2, eps=1e-6):
    """Fréchet distance between Gaussians fitted to two sets of features (rows), as pytorch-fid
    computes it."""
    from scipy import linalg

    f1, f2 = np.asarray(f1, np.float64), np.asarray(f2, np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.atleast_2d(np.cov(f1, rowvar=False)), np.atleast_2d(np.cov(f2, rowvar=False))
    covmean = linalg.sqrtm(s1.dot(s2))
    covmean = covmean[0] if isinstance(covmean, tuple) else covmean
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
        covmean = covmean[0] if isinstance(covmean, tuple) else covmean
    covmean = covmean.real
    d = mu1 - mu2
    return float(d.dot(d) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


class StyleScorer:
    """HWD, FID and KID of the HWD package, built once (their networks are downloaded on first
    use). height 32, as the package's defaults and its paper. FID is the Fréchet distance of
    the package's Inception features, computed here (the package's own call to
    scipy.linalg.sqrtm uses an argument that recent SciPy no longer accepts)."""

    def __init__(self, height=32):
        from hwd.scores import FIDScore, HWDScore, KIDScore     # noqa: F401  (fails early if missing)

        self.height = height
        self._made, self._reals = {}, {}

    def _get(self, name):
        if name not in self._made:
            import hwd.scores as s
            self._made[name] = {"hwd": s.HWDScore, "fid": s.FIDScore, "kid": s.KIDScore}[name](height=self.height)
        return self._made[name]

    def __call__(self, fake_root, real_root, which=("hwd",)):
        """Folders of one sub-folder per writer -> {name: score}. HWD compares the writers of
        fake_root with the same writers of real_root and averages over them."""
        from hwd.datasets import FolderDataset

        out = {}
        for name in which:
            score = self._get(name)
            key = (name, str(Path(real_root).resolve()))
            if key not in self._reals:            # the real words' features serve every setting
                self._reals[key] = score.digest(FolderDataset(str(real_root)))
            fake, real = score.digest(FolderDataset(str(fake_root))), self._reals[key]
            if name == "fid":
                out[name] = frechet(fake.features.cpu().numpy(), real.features.cpu().numpy())
            else:
                out[name] = float(score.distance(fake, real))
            if name == "hwd":                       # the same distance for each writer alone
                out["hwd_per_writer"] = {str(a): float(score.distance(fake[a], real[a]))
                                         for a in sorted(set(fake.authors))}
        return out


def choose(rows, reference_cer, margin=0.01, cer="greedy_cer", style="hwd"):
    """The setting the protocol picks among rows (dicts with the greedy CER and HWD)."""
    ok = [r for r in rows if r[cer] <= reference_cer + margin and r.get(style) is not None]
    if ok:
        return min(ok, key=lambda r: (r[style], r[cer]))
    return min(rows, key=lambda r: (r[cer], r.get(style) if r.get(style) is not None else float("inf")))


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def bootstrap(refs, hyps, n=2000, seed=0):
    """95% bootstrap intervals (resampling words) of CER and WER -> dict."""
    e = np.array([edit_distance(r, h) for r, h in zip(refs, hyps)], np.float64)
    c = np.array([len(r) for r in refs], np.float64)
    w = np.array([r != h for r, h in zip(refs, hyps)], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(refs), size=(n, len(refs)))
    cer = e[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    wer = w[idx].mean(1)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]
    return {"cer": round(float(e.sum() / max(c.sum(), 1)), 5), "cer_95ci": q(cer),
            "wer": round(float(w.mean()), 5), "wer_95ci": q(wer), "words": len(refs), "resamples": n}


def per_writer(predictions, writers):
    """Errors of each writer from a --predictions file of eval_recogniser.py, whose rows follow
    `writers` (the writer of each word, in order) -> {writer: {characters, char_errors (greedy and
    with_lm), words, word_errors (greedy and with_lm)}}, the counts a paired comparison needs."""
    rows = [line.split("\t") for line in Path(predictions).read_text(encoding="utf-8").splitlines()[1:] if line]
    out = {}
    for w, r in zip(writers, rows):
        ref, hyps = r[1], {"greedy": r[3], "with_lm": r[4] if len(r) > 4 else ""}
        d = out.setdefault(str(w), {"characters": 0, "words": 0, "char_errors_greedy": 0, "word_errors_greedy": 0,
                                    "char_errors_with_lm": 0, "word_errors_with_lm": 0})
        d["characters"] += len(ref)
        d["words"] += 1
        for k, h in hyps.items():
            d[f"char_errors_{k}"] += edit_distance(ref, h)
            d[f"word_errors_{k}"] += int(ref != h)
    return out


def paired_bootstrap(a, b, metric, n=5000, seed=0):
    """Two models scored on the same writers and words -> the difference b - a of a pooled error
    rate (metric "cer_greedy", "cer_with_lm", "wer_greedy", "wer_with_lm") or of the mean per-writer
    HWD ("hwd"), with a 95% interval from resampling writers."""
    writers = sorted(set(a) & set(b))
    if len(writers) < 2 or set(a) != set(b):
        raise ValueError("the two results need the same writers")
    if metric == "hwd":
        num = np.array([[b[w], a[w]] for w in writers], np.float64)
        den = np.ones(len(writers))
    else:
        rate, kind = metric.split("_", 1)
        err, tot = ("char_errors_" + kind, "characters") if rate == "cer" else ("word_errors_" + kind, "words")
        if any(a[w][tot] != b[w][tot] for w in writers):
            raise ValueError("the two results were scored on different words")
        num = np.array([[b[w][err], a[w][err]] for w in writers], np.float64)
        den = np.array([a[w][tot] for w in writers], np.float64)
    diff = num[:, 0] - num[:, 1]
    idx = np.random.default_rng(seed).integers(0, len(writers), size=(n, len(writers)))
    draws = diff[idx].sum(1) / den[idx].sum(1)
    return {"metric": metric, "b_minus_a": float(diff.sum() / den.sum()), "writers": len(writers), "resamples": n,
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def read_predictions(path, column="with_lm"):
    """A --predictions file of eval_recogniser.py -> (references, hypotheses)."""
    rows = [line.split("\t") for line in Path(path).read_text(encoding="utf-8").splitlines()[1:] if line]
    k = {"greedy": 3, "with_lm": 4}[column]
    return [r[1] for r in rows], [r[k] if len(r) > k else "" for r in rows]
