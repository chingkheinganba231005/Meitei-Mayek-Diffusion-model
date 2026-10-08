import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mayek_diffusion.model import Generator, ModelConfig
from mayek_diffusion.owner import split
from mayek_diffusion.recognition import draw_texts, finetune_recogniser, real_set, write_generated

ROOT = Path(__file__).resolve().parents[1]
TINY = ModelConfig(channels=(32, 64, 64), d=64, text_layers=1, style_layers=1)


@pytest.mark.slow
def test_generated_words_for_the_recogniser(word_repo, tools, recogniser_dir, tmp_path):
    from mayek_htr.data import FixedSet
    from mayek_htr.train import load_checkpoint

    synth, words, printer = tools
    texts = draw_texts(words.lexicon, 6, seed=300, printer=printer)
    assert len(texts) == 6 and texts == draw_texts(words.lexicon, 6, seed=300, printer=printer)
    torch.manual_seed(0)
    refs = [np.zeros((64, 40), np.float32)]
    refs[0][28:46, 4:36] = 1.0
    write_generated(Generator(TINY), texts, refs, printer, tmp_path / "gen.tar", batch=4, steps=2, log=lambda *a: None)
    gen = FixedSet(tmp_path / "gen.tar")
    assert gen.texts == texts and all(im.shape[0] == 64 for im in gen.images)
    out = tmp_path / "mixed.pt"
    h = finetune_recogniser(recogniser_dir / "recogniser.pt", words, out, generated=tmp_path / "gen.tar", steps=2,
                            batch=4, warmup=1, workers=0, device="cpu")
    assert len(h) == 1 and np.isfinite(h[0]["loss"])
    model, ck = load_checkpoint(out)
    assert ck["finetune"]["mix"] == 0.5 and ck["step"] == 2
    s = split(100)
    names = [line.split("\t")[0] for line in (word_repo / "real_words" / "labels.tsv").read_text(
        encoding="utf-8").splitlines() if line]
    s["files"] = {"references": [names[i] for i in s["references"]]}
    (tmp_path / "split.json").write_text(json.dumps(s))
    assert real_set(word_repo / "real_words", s["files"]["references"], tmp_path / "real.tar") == 96
    r = subprocess.run([sys.executable, "scripts/recognition.py", "score", "--word-repo", str(word_repo),
                        "--recogniser", str(recogniser_dir), "--models", f"start={recogniser_dir / 'recogniser.pt'}",
                        f"mixed={out}", "--real", str(word_repo / "real_words"), "--split", str(tmp_path / "split.json"),
                        "--synth-test", str(tmp_path / "gen.tar"), "--work", str(tmp_path / "work"),
                        "--out", str(tmp_path / "e6.json"), "--device", "cpu"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((tmp_path / "e6.json").read_text())
    assert res["real_words"] == 96 and set(res["models"]) == {"start", "mixed"}
    row = res["models"]["mixed"]["real"]
    assert row["greedy"]["words"] == 96 and len(row["with_lm_bootstrap"]["cer_95ci"]) == 2
