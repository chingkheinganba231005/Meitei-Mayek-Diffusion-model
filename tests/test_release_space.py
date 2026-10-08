import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mayek_diffusion.canvas import CANVAS_H, L_PX, LEFT, ink_columns
from mayek_diffusion.cards import cards, results_text
from mayek_diffusion.demo import (compose_line, enlarge, keyboard, load_presets, parse, references_from_images,
                                  save_canvases)
from mayek_diffusion.model import Generator, ModelConfig, config_dict
from mayek_diffusion.owner import prepare
from mayek_diffusion.protocol import BASE

ROOT = Path(__file__).resolve().parents[1]
TINY = ModelConfig(channels=(32, 64, 64), d=64, text_layers=1, style_layers=1)


def test_keyboard_and_text():
    keys = [c for _, cs in keyboard() for c in cs]
    assert len(keys) == len(set(keys)) == 54 and "ꯢ" not in keys and "꯫" in keys and "꯭" in keys
    assert parse(" ꯃꯤꯇꯩ  ꯃꯌꯦꯛ ") == ["ꯃꯤꯇꯩ", "ꯃꯌꯦꯛ"]
    assert parse("ꯑꯢ") == ["ꯑꯏ"]                       # everyday spelling
    for bad in ("", "   ", "abc", "ꯃ " * 7, "ꯃ" * 21):
        with pytest.raises(ValueError):
            parse(bad)


def test_line_and_presets(word_repo, tmp_path):
    a, b = np.zeros((CANVAS_H, 60), np.float32), np.zeros((CANVAS_H, 40), np.float32)
    a[30:46, 5:50], b[30:46, 5:30] = 1.0, 1.0
    line = compose_line([a, b])
    gap = int(round(0.5 * L_PX))
    assert line.shape == (CANVAS_H, LEFT + 45 + gap + 25 + LEFT)
    assert ink_columns(line) == (LEFT, LEFT + 45 + gap + 25)
    assert enlarge(line).shape == (3 * CANVAS_H, 3 * line.shape[1]) and enlarge(line).dtype == np.uint8
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in sorted((word_repo / "real_words" / "images").glob("*.png"))[:3]]
    refs, notes = references_from_images(imgs + [np.full((50, 80), 255, np.uint8)])
    assert len(refs) == 3 and len(notes) == 1 and all(r.shape[0] == CANVAS_H for r in refs)
    save_canvases(refs, tmp_path / "p" / "author")
    back = load_presets(tmp_path / "p")["author"]
    assert len(back) == 3 and np.abs(back[0] - refs[0]).max() <= 1 / 255 + 1e-6


def test_cards_fill_in():
    config = {"parameters": 113.5e6, "source": {"step": 1234}, "sampling": {"owner": "adapted"}}
    results = {"eval_test": {"writers": 1000, "words": 12000, "greedy_cer": 0.03, "with_lm_cer": 0.02,
                             "reference_greedy_cer": 0.01, "hwd": 1.2, "control": {"hwd": 2.5}, "fid": 10.0,
                             "kid": 0.01},
               "eval_val_grid": {"chosen": dict(BASE)}}
    text = results_text(results)
    assert "3.00%" in text and "2.500" in text and "relative width 0.1" in text and "4 reference words" in text
    assert "no guidance" in results_text({"eval_val_grid": {"chosen": dict(BASE, guidance=1.0, refs=1), "writers": 7}})
    model, space = cards(results, config)
    assert "113.5 M parameters" in model and "1234 steps" in model and "owner_generator" in model
    assert "{" not in space.split("---")[2] and space.startswith("---\ntitle:")
    assert "consent" in model and "consent" in space


@pytest.mark.slow
def test_export_and_space(word_repo, data16, tmp_path):
    torch.manual_seed(0)
    sd = Generator(TINY).state_dict()
    torch.save({"ema": {0.05: sd, 0.1: sd, 0.15: sd}, "model_cfg": config_dict(TINY), "cfg": {"prior": True},
                "step": 6}, tmp_path / "final.pt")
    torch.save({"model": sd, "model_cfg": config_dict(TINY), "cfg": {"prior": True}, "adapted": {"sigma_rel": 0.1}},
               tmp_path / "owner.pt")
    data = tmp_path / "data"
    data.mkdir()
    os.symlink(data16 / "test", data / "test")
    from mayek_diffusion.writers import Printer
    from mayek_words.synth import load_priors
    prepare(word_repo / "real_words", data / "owner", Printer(load_priors()))
    res = tmp_path / "results"
    res.mkdir()
    (res / "eval_val_grid.json").write_text(json.dumps({"chosen": dict(BASE, sigma_rel=0.15, guidance=1.5)}))
    rec = lambda cer: {"greedy": {"cer": cer}}
    (res / "owner.json").write_text(json.dumps({"held_out": 30, "real": rec(0.05),
                                                "few_shot": {"recognised": rec(0.06), "hwd": 2.0},
                                                "adapted": {"recognised": rec(0.055), "hwd": 1.0}}))
    r = subprocess.run([sys.executable, "scripts/export.py", "--word-repo", str(word_repo), "--weights",
                        str(tmp_path / "final.pt"), "--results", str(res), "--data", str(data), "--owner-weights",
                        str(tmp_path / "owner.pt"), "--out", str(tmp_path / "release")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    model_dir, space_dir = tmp_path / "release" / "model", tmp_path / "release" / "space"
    config = json.loads((model_dir / "config.json").read_text())
    assert config["sampling"]["sigma_rel"] == 0.15 and config["sampling"]["guidance"] == 1.5
    assert config["sampling"]["owner"] == "adapted" and (model_dir / "owner_generator.safetensors").exists()
    assert (space_dir / "mayek_words" / "assets" / "NotoSansMeeteiMayek-Regular.ttf").exists()
    assert (space_dir / "presets" / "author" / "4.png").exists() and (space_dir / "app.py").exists()
    assert "{hf_id}" not in (model_dir / "README.md").read_text()
    pytest.importorskip("gradio")
    check = (
        "import app\n"
        "words = ['ꯃꯤꯇꯩ', 'ꯃꯌꯦꯛ']\n"
        "image, path, notes = app.write(' '.join(words), app.AUTHOR, None, None, 2.0, 2, 1.0, 0, 2)\n"
        "assert image.shape[0] == 3 * 2 * (64 + 6) and app.OWNER is not None\n"
        "image, path, notes = app.write('ꯃꯤꯇꯩ', app.RANDOM, None, None, 2.0, 2, 1.0, 0, 1)\n"
        "image, path, notes = app.write('ꯃꯤꯇꯩ', app.PRESET, list(app.PRESETS)[0], None, 2.0, 2, 0.8, 3, 1)\n"
        "assert image.shape[0] == 3 * (64 + 6)\n"
        "gallery, msg = app.show_refs(app.PRESET, list(app.PRESETS)[0], None)\n"
        "assert len(gallery) == 4\n"
        "app.build()\n"
        "print('space ok')\n")
    env = dict(os.environ, MODEL_DIR=str(model_dir), THREADS="2")
    r = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, cwd=space_dir, env=env)
    assert r.returncode == 0 and "space ok" in r.stdout, r.stderr[-3000:]
