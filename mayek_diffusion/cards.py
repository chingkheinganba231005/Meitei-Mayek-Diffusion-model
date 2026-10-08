"""The model card and the Space card, filled in from the results files.

The templates are space/MODEL_CARD.md and space/README.md; their {fields} are filled here.
A result that does not exist yet is left out of the text.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/chingkheinganba231005/Meitei-Mayek-Diffusion-model"
HF_ID = "Chingkheinganba/handwritten-meitei-mayek-word-generation"
FILES = ("eval_test.json", "eval_val_grid.json", "owner.json", "data_stats.json", "speed_cpu.json")


def load_results(folder):
    """{name without .json: contents} of the results files that exist."""
    out = {}
    for f in FILES:
        p = Path(folder) / f
        if p.exists():
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def pct(x, digits=2):
    return f"{100 * x:.{digits}f}%"


def setting_text(s):
    sampler = "Heun's method" if s["sampler"] == "heun" else "DPM-Solver++(2M)"
    lo, hi = s["interval"]
    where = "at every noise level" if hi >= 1000 else f"for noise levels in ({lo:g}, {hi:g}]"
    return (f"the average of the weights with relative width {s['sigma_rel']:g}, {sampler} with {s['steps']} steps, "
            f"guidance {s['guidance']:g} on the style {where}, {s['refs']} reference word(s)")


def results_text(r):
    """The paragraph of results for the cards."""
    parts = []
    if "data_stats" in r and "train" in r["data_stats"]:
        t = r["data_stats"]["train"]
        parts.append(f"It was trained on {t['words']:,} synthetic words of {t['writers']:,} pseudo-writers composed "
                     "from the isolated characters of TUMMHCD, without any real handwritten word.")
    if "eval_val_grid" in r and r["eval_val_grid"].get("chosen"):
        parts.append("Sampling settings chosen on 200 validation writers: " + setting_text(r["eval_val_grid"]["chosen"])
                     + ".")
    if "eval_test" in r:
        e = r["eval_test"]
        s = (f"On {e['writers']:,} test writers ({e['words']:,} generated words), the word recogniser reads the "
             f"generated words with a character error rate of {pct(e['greedy_cer'])} (greedy decoding; "
             f"{pct(e['with_lm_cer'])} with its language model), against {pct(e['reference_greedy_cer'])} on the "
             "real synthetic words of the same writers and texts.")
        if e.get("hwd") is not None:
            s += f" HWD to the writers' own words: {e['hwd']:.3f}"
            if e.get("control", {}).get("hwd") is not None:
                s += f" (the same texts in another writer's hand: {e['control']['hwd']:.3f})"
            s += "."
        if e.get("fid") is not None:
            s += f" FID {e['fid']:.2f}, KID {e['kid']:.4f}."
        parts.append(s)
    if "owner" in r:
        o = r["owner"]
        f, a = o["few_shot"], o["adapted"]
        s = (f"On {o['held_out']} held-out words of the author's own hand: with four of their words as references, "
             f"character error rate {pct(f['recognised']['greedy']['cer'])}")
        if f.get("hwd") is not None:
            s += f" and HWD {f['hwd']:.3f}"
        s += f"; after adaptation on 70 words, {pct(a['recognised']['greedy']['cer'])}"
        if a.get("hwd") is not None:
            s += f" and HWD {a['hwd']:.3f}"
        parts.append(s + f" (the real words: {pct(o['real']['greedy']['cer'])}).")
    if "speed_cpu" in r:
        s = r["speed_cpu"]
        parts.append(f"On {s['threads']} CPU threads a word takes about {s['seconds_per_word']:.0f} s "
                     f"({s['steps']} steps of DPM-Solver++(2M)).")
    return " ".join(parts)


def owner_text(config):
    use = config.get("sampling", {}).get("owner")
    if use == "adapted":
        return ("`owner_generator.safetensors` is the generator adapted to the author's hand on 70 of their words; "
                "the demo uses it for \"the author's hand\".")
    return ""


def cards(results, config, hf_id=HF_ID, licence="mit"):
    """(model card, Space card) as Markdown."""
    fill = {"hf_id": hf_id, "repo_url": REPO_URL, "results": results_text(results), "licence": licence,
            "parameters": f"{config['parameters'] / 1e6:.1f}", "step": config.get("source", {}).get("step", "?"),
            "owner": owner_text(config)}
    model = (ROOT / "space" / "MODEL_CARD.md").read_text(encoding="utf-8").format(**fill)
    space = (ROOT / "space" / "README.md").read_text(encoding="utf-8").format(**fill)
    return model, space


def results_json(results, config):
    """results.json of the model repository: the numbers the cards quote."""
    keep = ("writers", "words", "greedy_cer", "greedy_wer", "with_lm_cer", "with_lm_wer", "reference_greedy_cer",
            "reference_with_lm_cer", "hwd", "fid", "kid")
    out = {"setting": config["sampling"], "source": config.get("source")}
    if "eval_test" in results:
        e = results["eval_test"]
        out["test"] = {k: e.get(k) for k in keep}
        if "control" in e:
            out["test"]["control_hwd"] = e["control"].get("hwd")
        out["test"]["recogniser"] = e.get("recogniser")
    if "owner" in results:
        o = results["owner"]
        out["owner"] = {k: {"greedy_cer": o[k]["recognised"]["greedy"]["cer"], "hwd": o[k].get("hwd"),
                            "kid": o[k].get("kid")} for k in ("few_shot", "adapted")}
        out["owner"]["real_greedy_cer"] = o["real"]["greedy"]["cer"]
    return out
