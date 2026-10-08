"""Render a split of pseudo-writers to shards.

Every writer writes `words` words; each word is kept as its canvas (uint8, ink 255, paper 0)
and its printed prior stretched over the columns of the word's ink (same width). Writer w's
words are items w * words ... w * words + words - 1 of the split. A shard holds the writers
[k * shard_writers, (k + 1) * shard_writers) and is a compressed .npz:

    targets, priors  uint8 (64, total width): the canvases side by side
    offsets          int64 (n + 1): item i is columns offsets[i]:offsets[i + 1]
    texts            unicode (n,): the words as written (everyday spelling)
    writer           int32 (n,)
    ink              int16 (n, 2): first and last + 1 column of the word's ink
    print_width      int16 (n,): width of the printed word's ink before stretching
    style            float32 (n, len(STYLE_KEYS)): the word's style, for analyses

Everything random comes from numpy Generators seeded with (seed, writer), so a shard is
the same whichever process renders it.
"""

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .canvas import CANVAS_H, ink_columns, stretch, to_uint8
from .writers import Printer, WriterSynth, from_sample, writer_config

MAX_WIDTH = 384
STYLE_KEYS = ("L", "width", "gap", "mark_scale", "slant", "pen", "touch", "uneven", "blur", "marks_beside")
SPLIT_SEEDS = {"train": 100, "val": 101, "test": 102}

_STATE = {}


def _setup(glyphs, lexicon, sizes, allographs, scrambled, overrides):
    from mayek_words.glyphs import GlyphStore
    from mayek_words.lexicon import Lexicon
    from mayek_words.synth import Words, load_priors

    priors = load_priors(sizes=sizes)
    synth = WriterSynth(GlyphStore.load(glyphs), priors, writer_config(**overrides), allographs=allographs)
    _STATE.update(synth=synth, printer=Printer(priors),
                  words=Words(synth, Lexicon.load(lexicon), seed=0, scrambled=scrambled))


def render_writers(writers, n_words, seed, synth=None, words=None, printer=None, tries=50):
    """The words of the given writers -> dict of arrays (as a shard)."""
    synth = synth or _STATE["synth"]
    words = words or _STATE["words"]
    printer = printer or _STATE["printer"]
    cols_t, cols_p, offsets, texts, wid, ink, pw, style = [], [], [0], [], [], [], [], []
    for w in writers:
        rng = np.random.default_rng([seed, int(w)])
        synth.new_writer(rng)
        for _ in range(n_words):
            for _ in range(tries):
                try:
                    s = synth.render(words.text(rng), rng)
                except (ValueError, AssertionError):
                    continue
                x, lost = from_sample(s)
                c = ink_columns(x)
                if c is not None and lost <= 0.02 and x.shape[1] <= MAX_WIDTH:
                    break
            else:
                raise RuntimeError(f"writer {w}: no word could be drawn in {tries} tries")
            p = printer(s.text)
            pc = ink_columns(p)
            cols_t.append(to_uint8(x))
            cols_p.append(to_uint8(stretch(p, c[0], c[1], x.shape[1])))
            offsets.append(offsets[-1] + x.shape[1])
            texts.append(s.text)
            wid.append(w)
            ink.append(c)
            pw.append(pc[1] - pc[0])
            style.append([float(s.style[k] or 0.0) for k in STYLE_KEYS])
    return {"targets": np.concatenate(cols_t, 1), "priors": np.concatenate(cols_p, 1),
            "offsets": np.asarray(offsets, np.int64), "texts": np.asarray(texts),
            "writer": np.asarray(wid, np.int32), "ink": np.asarray(ink, np.int16),
            "print_width": np.asarray(pw, np.int16), "style": np.asarray(style, np.float32)}


def _shard(job):
    k, writers, n_words, seed, path = job
    if Path(path).exists():
        return path, 0.0
    t0 = time.time()
    arrays = render_writers(writers, n_words, seed)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)
    return path, time.time() - t0


def render_split(split, glyphs, lexicon, sizes, out_dir, writers, words=16, shard_writers=2000, processes=None,
                 allographs=2, scrambled=0.1, seed=None, **overrides):
    """Render `writers` writers of a split to out_dir/<split>/shard_*.npz (shards already there
    are kept, so an interrupted run resumes) and write out_dir/<split>/info.json."""
    seed = SPLIT_SEEDS[split] if seed is None else seed
    out = Path(out_dir) / split
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, range(a, min(a + shard_writers, writers)), words, seed, out / f"shard_{k:04d}.npz")
            for k, a in enumerate(range(0, writers, shard_writers))]
    args = (str(glyphs), str(lexicon), str(sizes), allographs, scrambled, overrides)
    t0, before = time.time(), {}
    if (out / "info.json").exists():                 # an earlier session: keep its minutes
        before = json.loads((out / "info.json").read_text(encoding="utf-8"))
    times_path = out / "times.json"         # seconds of one process per shard, kept across sessions
    times = json.loads(times_path.read_text(encoding="utf-8")) if times_path.exists() else {}
    with mp.get_context("fork").Pool(processes, initializer=_setup, initargs=args) as pool:
        for path, secs in pool.imap_unordered(_shard, jobs):
            print(f"{Path(path).name}: {secs:.0f} s", flush=True)
            if secs:
                times[Path(path).name] = round(secs, 1)
                times_path.write_text(json.dumps(times, indent=1), encoding="utf-8")
    info = {"split": split, "writers": writers, "words_per_writer": words, "shard_writers": shard_writers,
            "seed": seed, "allographs": allographs, "scrambled": scrambled, "overrides": overrides,
            "glyphs": str(glyphs), "lexicon": str(lexicon), "sizes": str(sizes), "max_width": MAX_WIDTH,
            "style_keys": STYLE_KEYS, "shards": [Path(j[4]).name for j in jobs],
            "minutes": round(before.get("minutes", 0.0) + (time.time() - t0) / 60, 1),
            "core_minutes": round(sum(times.values()) / 60, 1)}
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def stats(out_dir, splits=("train", "val", "test")):
    """Counts of the rendered splits (read shard by shard, without the images) -> dict."""
    above = set("ꯥꯩꯪ")
    out = {}
    for split in splits:
        folder = Path(out_dir) / split
        if not (folder / "info.json").exists():
            continue
        info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
        widths, ink, lengths, beside, has_above, writers, size = [], [], [], [], [], [], 0
        k_beside = list(info["style_keys"]).index("marks_beside")
        for name in info["shards"]:
            size += (folder / name).stat().st_size
            with np.load(folder / name) as z:
                widths.append(np.diff(z["offsets"]))
                ink.append(z["ink"][:, 1] - z["ink"][:, 0])
                texts = z["texts"]
                lengths.append(np.array([len(t) for t in texts]))
                has_above.append(np.array([any(c in above for c in t) for t in texts]))
                beside.append(z["style"][:, k_beside] > 0.5)
                writers.append(z["writer"])
        widths, ink, lengths = np.concatenate(widths), np.concatenate(ink), np.concatenate(lengths)
        beside, has_above, writers = np.concatenate(beside), np.concatenate(has_above), np.concatenate(writers)
        pct = lambda v: {str(q): float(np.percentile(v, q)) for q in (50, 90, 99)} | {"max": float(v.max())}
        out[split] = {"writers": int(len(np.unique(writers))), "words": int(len(widths)),
                      "words_per_writer": info["words_per_writer"], "seed": info.get("seed"),
                      "canvas_width": pct(widths), "ink_width": pct(ink), "characters": pct(lengths),
                      "writers_signs_beside": round(float(beside[::info["words_per_writer"]].mean()), 4),
                      "words_with_sign_above": round(float(has_above.mean()), 4),
                      "words_with_sign_above_written_beside": round(float((has_above & beside).mean()), 4),
                      "core_minutes": info.get("core_minutes"), "megabytes": round(size / 1e6, 1)}
    return out
