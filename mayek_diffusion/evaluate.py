"""Generated test sets, written as the scoring tools read them.

Test writers come from the test split (TUMMHCD test characters, test words). For each writer
the first `refs` words are its references and the generator writes the texts of its other
words, so that every generated word has a real word of the same writer and text beside it.
- write_fixed_set: a .tar that the recogniser's scripts/eval_recogniser.py scores (CER, WER):
  images cut to their ink with a margin of 0.15 of its height, dark on white, as the real
  words were cut.
- write_writer_folders: a folder per writer of such images (the layout HWD reads).
"""

import io
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from .canvas import crop_ink, ink_columns
from .sample import generate, paper


def generate_writers(model, ds, printer, writers, refs=4, batch=64, seed=0, first=None, words=None, ref_of=None,
                     **kw):
    """-> list of dicts (writer, text, generated canvas, real canvas) over the given writers, in
    order. Words of several writers share a batch; batches group words of similar printed width.
    The generated words are items first .. first + words - 1 of each writer (default: all after
    the references). ref_of(w): the writer whose references are used for writer w (the
    wrong-style control); by default w itself."""
    per, jobs = ds.per_writer, []
    first = refs if first is None else first
    words = per - first if words is None else words
    assert refs <= first and first + words <= per, "references and generated words overlap"
    for w in writers:
        items = list(range(int(w) * per, (int(w) + 1) * per))
        rw = int(ref_of(w)) if ref_of is not None else int(w)
        ref = [ds.target(i).astype(np.float32) / 255 for i in range(rw * per, rw * per + refs)]
        jobs += [(int(w), i, ref) for i in items[first:first + words]]
    order = sorted(range(len(jobs)), key=lambda j: ink_width(printer(ds.texts[jobs[j][1]])))
    gen = [None] * len(jobs)
    for k in range(0, len(order), batch):
        chunk = order[k:k + batch]
        out = generate(model, [ds.texts[jobs[j][1]] for j in chunk], [jobs[j][2] for j in chunk], printer,
                       seed=seed + k, **kw)
        for j, g in zip(chunk, out):
            gen[j] = g
    return [{"writer": w, "text": ds.texts[i], "generated": g, "real": ds.target(i).astype(np.float32) / 255}
            for (w, i, _), g in zip(jobs, gen)]


def ink_width(canvas):
    c = ink_columns(canvas)
    return 0 if c is None else c[1] - c[0]


def word_png(canvas):
    """Float canvas -> PNG bytes of the word cut to its ink, dark on white."""
    buf = io.BytesIO()
    Image.fromarray(crop_ink(paper(canvas))).save(buf, "PNG")
    return buf.getvalue()


def write_fixed_set(canvases, texts, path):
    """images/000001.png ... and labels.tsv in a .tar (mayek_htr.data.FixedSet)."""
    rows = []
    with tarfile.open(path, "w") as tar:
        for k, (c, t) in enumerate(zip(canvases, texts), 1):
            data, name = word_png(c), f"{k:06d}.png"
            info = tarfile.TarInfo("images/" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            rows.append(f"{name}\t{t}")
        data = ("\n".join(rows) + "\n").encode("utf-8")
        info = tarfile.TarInfo("labels.tsv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def write_writer_folders(canvases, writers, root):
    """root/<writer>/<k>.png."""
    root = Path(root)
    for k, (c, w) in enumerate(zip(canvases, writers)):
        d = root / f"{w:05d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{k:06d}.png").write_bytes(word_png(c))
