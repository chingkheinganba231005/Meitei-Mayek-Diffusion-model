"""The released generator, and the weights of a training run.

A release is a folder with `generator.safetensors` (one average of the weights, float16) and
`config.json`: the model's configuration, the diffusion settings, the canvas, the alphabet and
the sampling settings chosen on validation, so that the generator can be rebuilt and used
without this repository's training code. The adapted generator (the owner's hand) is saved in
the same way as `owner_generator.safetensors`, sharing config.json.

`load_weights` reads any of the files a run leaves: final.pt (the three averages), a
snapshot snap_*.pt (the 0.10 average in bfloat16), last.pt, the adapted owner.pt, or a
release folder.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from mayek_words.charset import ALPHABET

from .canvas import BASE_ROW, CANVAS_H, L_PX, LEFT, MULTIPLE
from .model import Generator, ModelConfig, config_dict, count
from .render import MAX_WIDTH

FORMAT = "mayek-diffusion/1"
# the sampling defaults before tuning (Section 9 of the plan chooses them on validation)
SAMPLING = {"sigma_rel": 0.10, "guidance": 2.0, "interval": [0.3, 5.0], "refs": 4, "sampler": "heun", "steps": 18,
            "demo_sampler": "dpmpp", "demo_steps": 14, "width_scale": [0.8, 1.25]}


def model_config(d):
    """ModelConfig from its dict (lists back to tuples)."""
    return ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def _floats(state, dtype=torch.float32):
    return {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()}


def load_weights(path, sigma_rel=0.10, device="cpu"):
    """A generator from a file of a run or a release folder -> (Generator in eval mode, info).
    info: kind, step, sigma_rel, prior (whether the model was trained with the printed prior;
    None when the file does not say)."""
    path = Path(path)
    if path.is_dir():
        name = "owner_generator" if (path / "owner_generator.safetensors").exists() and \
            not (path / "generator.safetensors").exists() else "generator"
        model, config = load(path, name, device)
        return model, {"kind": "release", "step": config.get("source", {}).get("step"),
                       "sigma_rel": config["sampling"].get("sigma_rel"), "prior": config.get("prior", True)}
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or {}
    info = {"step": ck.get("step"), "prior": cfg.get("prior", ck.get("prior")), "sigma_rel": None}
    if "adapted" in ck:                                   # finetune.adapt
        state, info["kind"], info["sigma_rel"] = ck["model"], "adapted", ck["adapted"].get("sigma_rel")
    elif isinstance(ck.get("ema"), dict) and "models" in ck["ema"]:          # last.pt
        rels = [float(s) for s in ck["ema"]["sigma_rels"]]
        state, info["kind"], info["sigma_rel"] = ck["ema"]["models"][rels.index(float(sigma_rel))], "last", sigma_rel
    elif isinstance(ck.get("ema"), dict) and any(isinstance(k, float) for k in ck["ema"]):   # final.pt
        state, info["kind"], info["sigma_rel"] = ck["ema"][float(sigma_rel)], "final", sigma_rel
    elif "ema" in ck:                                     # snap_*.pt: the 0.10 average
        state, info["kind"], info["sigma_rel"] = ck["ema"], "snapshot", 0.10
    else:
        raise ValueError(f"{path}: not a file of a training run")
    model = Generator(model_config(ck["model_cfg"]))
    model.load_state_dict(_floats(state))
    return model.to(device).eval(), info


def export(model, out_dir, name="generator", sampling=None, edm=None, prior=True, source=None):
    """Write <name>.safetensors (float16) and config.json to out_dir -> the two paths. An existing
    config.json is kept if it describes the same model (the adapted generator shares it)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu", torch.float16 if v.is_floating_point() else v.dtype).contiguous()
             for k, v in model.state_dict().items()}
    weights = out / f"{name}.safetensors"
    save_file(state, weights, metadata={"format": FORMAT})
    config = {"format": FORMAT, "model": config_dict(model.cfg),
              "edm": edm or {"sigma_data": 0.5, "p_mean": -1.2, "p_std": 1.2},
              "canvas": {"height": CANVAS_H, "letter_height": L_PX, "baseline_row": BASE_ROW, "left": LEFT,
                         "multiple": MULTIPLE, "max_width": MAX_WIDTH},
              "alphabet": "".join(ALPHABET), "prior": bool(prior), "sampling": {**SAMPLING, **(sampling or {})},
              "parameters": count(model), "source": source or {}}
    path = out / "config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("model") != config["model"]:
            raise ValueError(f"{path} describes another model")
        if name != "generator":
            return weights, path
    path.write_text(json.dumps(config, indent=1, ensure_ascii=False), encoding="utf-8")
    return weights, path


def load(folder, name="generator", device="cpu", dtype=torch.float32):
    """A release folder -> (Generator in eval mode, config dict). The float16 weights are
    loaded as `dtype` (float32 for CPU inference)."""
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != FORMAT:
        raise ValueError(f"{folder / 'config.json'}: unknown format {config.get('format')!r}")
    if config["alphabet"] != "".join(ALPHABET):
        raise ValueError("the release was made for another alphabet")
    model = Generator(model_config(config["model"]))
    model.load_state_dict(_floats(load_file(folder / f"{name}.safetensors"), dtype))
    return model.to(device=device, dtype=dtype).eval(), config


def pick_presets(ds, n=6, pool=200):
    """n writers of a rendered split for the demo: half writing ꯥ ꯩ ꯪ beside their letter, half
    as in print, spread over slant within each half -> writer indices."""
    from .render import STYLE_KEYS

    per = ds.per_writer
    style = np.concatenate([s["style"] for s in ds.shards])[::per][:pool]
    beside, slant = style[:, STYLE_KEYS.index("marks_beside")] > 0.5, style[:, STYLE_KEYS.index("slant")]
    out = []
    for group, k in ((np.flatnonzero(beside), n // 2), (np.flatnonzero(~beside), n - n // 2)):
        order = group[np.argsort(slant[group])]
        if len(order):
            out += [int(order[int(round(q * (len(order) - 1)))]) for q in np.linspace(0, 1, k)]
    return sorted(set(out))


def vendor(space_dir, word_repo):
    """Copy the code the Space needs next to app.py: this package and the word repository's
    mayek_words (with its assets and licence)."""
    import shutil

    space_dir = Path(space_dir)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for src, name in ((Path(__file__).resolve().parent, "mayek_diffusion"),
                      (Path(word_repo) / "mayek_words", "mayek_words")):
        shutil.rmtree(space_dir / name, ignore_errors=True)
        shutil.copytree(src, space_dir / name, ignore=ignore)
    shutil.copyfile(Path(word_repo) / "LICENSE", space_dir / "mayek_words" / "LICENSE")
