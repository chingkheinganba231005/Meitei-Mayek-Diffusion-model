"""The evaluation protocol: generated words of real synthetic writers, scored for legibility
and style, and the choice of the sampling setting on validation writers.

Each writer of a split has 16 words. With k references, the generator writes the texts of
the writer's other words (items `first` to `first + words - 1`; by default items 4-15, the
12 words after the first four), so that every generated word has the real word of the same
writer and text beside it. The reference level is the recogniser on those real words, cut and
processed in the same way.

The grid (E3) is run in stages, each choosing by the rule of scores.choose and handing its
choice to the next: the average of the weights; then guidance and its interval; then the
sampler and its steps; then the number of references (1, 2, 4, 8; the generated words are
then items 8-15 for every k, so that 8 references fit). Every row is written to the results
file as soon as it is scored, and a row already there is not run again, so an interrupted
grid resumes.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .edm import karras_sigmas
from .evaluate import generate_writers, write_fixed_set, write_writer_folders
from .release import load_weights
from .scores import choose
from .sheets import grid

BASE = {"sigma_rel": 0.10, "guidance": 2.0, "interval": [0.3, 5.0], "sampler": "heun", "steps": 18, "refs": 4}
ALL_SIGMAS = [0.0, 1000.0]          # guidance at every noise level
RULE = ("the lowest HWD among the settings whose greedy CER is at most the reference CER plus 1 percentage "
        "point; if none qualifies, the lowest greedy CER")


def stages(best=None):
    """The settings of each stage given the choices so far -> list of (name, settings)."""
    b = dict(BASE)
    out = [("average", [dict(b, sigma_rel=s) for s in (0.05, 0.10, 0.15)])]
    if best and "average" in best:
        b = dict(best["average"])
        out.append(("guidance", [dict(b, guidance=1.0, interval=ALL_SIGMAS)] +
                    [dict(b, guidance=g, interval=iv) for g in (1.5, 2.0, 3.0)
                     for iv in (ALL_SIGMAS, [0.3, 5.0], [0.1, 2.0])]))
    if best and "guidance" in best:
        b = dict(best["guidance"])
        out.append(("sampler", [dict(b, sampler="heun", steps=18)] +
                    [dict(b, sampler="dpmpp", steps=n) for n in (8, 12, 16, 24)]))
    if best and "sampler" in best:
        b = dict(best["sampler"])
        out.append(("references", [dict(b, refs=k) for k in (1, 2, 4, 8)]))
    return out


def targets(name):
    """(first, words) of the generated words of a stage."""
    return (8, 8) if name == "references" else (4, 12)


def key(setting, first, words):
    s = setting
    iv = "all" if s["interval"] == ALL_SIGMAS else f"{s['interval'][0]}-{s['interval'][1]}"
    return (f"avg{s['sigma_rel']}_g{s['guidance']}_i{iv}_{s['sampler']}{s['steps']}_r{s['refs']}"
            f"_w{first}-{first + words}")


def cost(setting):
    """Network evaluations of one word: (calls, of which guided; a guided call costs two)."""
    sig = karras_sigmas(setting["steps"])
    at = list(sig[:-1])
    if setting["sampler"] == "heun":
        at += [s for s in sig[1:] if s > 0]
    lo, hi = setting["interval"]
    guided = 0 if setting["guidance"] == 1.0 else sum(lo < s <= hi for s in at)
    return len(at), guided


class Evaluator:
    """Generates and scores the words of `writers` of a split (data.WordSet).

    recognise(fixed_set, out_json) -> {"words", "greedy", "with_lm"}; style(fake_root, real_root,
    which) -> {name: score} or None (no style scores). work: a local folder for the images."""

    def __init__(self, weights, ds, printer, writers, recognise, style, work, device="cuda", batch=64,
                 prior=None, log=print):
        self.weights, self.ds, self.printer, self.writers = weights, ds, printer, [int(w) for w in writers]
        self.recognise_fn, self.style_fn, self.work = recognise, style, Path(work)
        self.device, self.batch, self.log, self.prior = device, batch, log, prior
        self.dtype = torch.bfloat16 if str(device).startswith("cuda") else None
        self.models, self.refs_cache = {}, {}

    def model(self, sigma_rel):
        if sigma_rel not in self.models:
            self.models.clear()                  # one average in memory at a time
            m, info = load_weights(self.weights, sigma_rel, self.device)
            self.models[sigma_rel] = (m, info)
        return self.models[sigma_rel]

    def reference(self, first, words):
        """The recogniser on the real words that the generated words copy (once per range)."""
        k = (first, words)
        if k not in self.refs_cache:
            d = self.work / f"real_w{first}-{first + words}"
            per = self.ds.per_writer
            items = [w * per + i for w in self.writers for i in range(first, first + words)]
            canvases = [self.ds.target(i).astype(np.float32) / 255 for i in items]
            texts = [str(self.ds.texts[i]) for i in items]
            d.mkdir(parents=True, exist_ok=True)
            write_fixed_set(canvases, texts, d / "set.tar")
            if not (d / "writers").exists():
                write_writer_folders(canvases, [int(self.ds.writer[i]) for i in items], d / "writers")
            self.refs_cache[k] = {"folder": d / "writers", "scores": self.recognise_fn(d / "set.tar", d / "read.json")}
        return self.refs_cache[k]

    def run(self, setting, first=4, words=12, control=False, fid_kid=False, keep=False, seed=0):
        """Generate and score one setting -> row (dict). control: also the same texts written with
        the next writer's references (the wrong-style control)."""
        model, info = self.model(setting["sigma_rel"])
        prior = info.get("prior") if self.prior is None else self.prior
        prior = True if prior is None else bool(prior)
        ref = self.reference(first, words)
        name = key(setting, first, words)
        d = self.work / name
        kw = dict(refs=setting["refs"], first=first, words=words, batch=self.batch, seed=seed,
                  steps=setting["steps"], sampler=setting["sampler"], guidance=setting["guidance"],
                  interval=tuple(setting["interval"]), device=self.device, dtype=self.dtype, prior=prior)
        t0 = time.time()
        rows = generate_writers(model, self.ds, self.printer, self.writers, **kw)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.time() - t0
        calls, guided = cost(setting)
        out = {"key": name, "setting": setting, "first": first, "words": len(rows), "writers": len(self.writers),
               "nfe": calls, "nfe_guided": guided, "seconds_per_word": round(seconds / max(len(rows), 1), 4),
               **self._score(rows, d, ref, fid_kid),
               "reference_greedy_cer": ref["scores"]["greedy"]["cer"],
               "reference_with_lm_cer": ref["scores"]["with_lm"]["cer"] if ref["scores"]["with_lm"] else None}
        if control:
            nxt = {w: self.writers[(j + 1) % len(self.writers)] for j, w in enumerate(self.writers)}
            crow = generate_writers(model, self.ds, self.printer, self.writers, ref_of=lambda w: nxt[w], **kw)
            out["control"] = self._score(crow, self.work / (name + "_control"), ref, False)
        if not keep:
            shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(self.work / (name + "_control"), ignore_errors=True)
        self.last_rows = rows
        return out

    def _score(self, rows, d, ref, fid_kid):
        d.mkdir(parents=True, exist_ok=True)
        gen, texts = [r["generated"] for r in rows], [r["text"] for r in rows]
        write_fixed_set(gen, texts, d / "set.tar")
        s = self.recognise_fn(d / "set.tar", d / "read.json")
        out = {"greedy_cer": s["greedy"]["cer"], "greedy_wer": s["greedy"]["wer"],
               "with_lm_cer": s["with_lm"]["cer"] if s["with_lm"] else None,
               "with_lm_wer": s["with_lm"]["wer"] if s["with_lm"] else None, "recognised": s}
        if self.style_fn is not None:
            shutil.rmtree(d / "writers", ignore_errors=True)
            write_writer_folders(gen, [r["writer"] for r in rows], d / "writers")
            out.update(self.style_fn(d / "writers", ref["folder"], ("hwd", "fid", "kid") if fid_kid else ("hwd",)))
        return out


def load_rows(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def tune(ev, out_path, extra=None, log=print):
    """Run the staged grid with resume -> the results (also written to out_path)."""
    res = load_rows(out_path) or {"rule": RULE, "rows": {}, "stages": {}}
    res.update(extra or {})
    res["writers"], res["base"] = len(ev.writers), BASE
    best = {}

    def save():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    while True:
        todo = [s for s in stages(best) if s[0] not in best]
        if not todo:
            break
        name, settings = todo[0]
        first, words = targets(name)
        rows = []
        for st in settings:
            k = key(st, first, words)
            if k not in res["rows"]:
                log(f"{name}: {k}")
                res["rows"][k] = ev.run(st, first, words)
                save()
            r = res["rows"][k]
            log(f"  {k}: greedy CER {r['greedy_cer']:.4f}, HWD {r.get('hwd')}")
            rows.append(r)
        ref_cer = rows[0]["reference_greedy_cer"]
        chosen = choose(rows, ref_cer)
        best[name] = chosen["setting"]
        res["stages"][name] = {"keys": [r["key"] for r in rows], "reference_greedy_cer": ref_cer,
                               "chosen": chosen["key"]}
        save()
    res["chosen"] = best["references"]
    save()
    return res


def chosen_setting(path_or_dict):
    """The chosen setting of a grid results file (or BASE when there is none)."""
    if path_or_dict is None:
        return dict(BASE)
    d = path_or_dict if isinstance(path_or_dict, dict) else load_rows(path_or_dict)
    return dict(d.get("chosen") or BASE)


def writer_sheet(ev, rows, path, writers=8, show=3, refs=4):
    """references | real | generated for the first writers of the rows."""
    per, lines = ev.ds.per_writer, []
    by_writer = {}
    for r in rows:
        by_writer.setdefault(r["writer"], []).append(r)
    for w in list(by_writer)[:writers]:
        ref = [ev.ds.target(i).astype(np.float32) / 255 for i in range(w * per, w * per + refs)]
        rs = by_writer[w][:show]
        lines.append([ref, [r["real"] for r in rs], [r["generated"] for r in rs]])
    return grid(lines, path)
