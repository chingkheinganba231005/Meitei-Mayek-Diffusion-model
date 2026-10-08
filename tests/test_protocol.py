import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from mayek_diffusion import protocol
from mayek_diffusion.data import WordSet
from mayek_diffusion.model import Generator, ModelConfig, config_dict
from mayek_diffusion.scores import bootstrap, choose, edit_distance, read_predictions, recognise

TINY = ModelConfig(channels=(32, 64, 64), d=64, text_layers=1, style_layers=1)
ROOT = Path(__file__).resolve().parents[1]


def final_pt(path, prior=True):
    torch.manual_seed(0)
    sd = Generator(TINY).state_dict()
    torch.save({"ema": {0.05: sd, 0.1: sd, 0.15: sd}, "model_cfg": config_dict(TINY), "cfg": {"prior": prior},
                "step": 6}, path)
    return path


def test_choose_rule():
    rows = [{"greedy_cer": 0.05, "hwd": 1.0, "k": "a"}, {"greedy_cer": 0.02, "hwd": 2.0, "k": "b"},
            {"greedy_cer": 0.30, "hwd": 0.1, "k": "c"}]
    assert choose(rows, 0.045)["k"] == "a"           # within 1 point of the reference: lowest HWD
    assert choose(rows, 0.005)["k"] == "b"          # none within: lowest CER
    assert choose([dict(r, hwd=None) for r in rows], 0.045)["k"] == "b"


def test_stages_and_costs():
    names = [n for n, _ in protocol.stages()]
    assert names == ["average"]
    best = {"average": dict(protocol.BASE, sigma_rel=0.05)}
    g = dict(protocol.stages(best))["guidance"]
    assert len(g) == 10 and all(s["sigma_rel"] == 0.05 for s in g)
    best["guidance"] = g[3]
    best["sampler"] = dict(protocol.stages(best))["sampler"][2]
    refs = dict(protocol.stages(best))["references"]
    assert [s["refs"] for s in refs] == [1, 2, 4, 8] and protocol.targets("references") == (8, 8)
    assert protocol.cost(dict(protocol.BASE))[0] == 35 and protocol.cost(dict(protocol.BASE, sampler="dpmpp"))[0] == 18
    assert protocol.cost(dict(protocol.BASE, guidance=1.0))[1] == 0
    calls, guided = protocol.cost(dict(protocol.BASE, interval=protocol.ALL_SIGMAS))
    assert guided == calls


def test_bootstrap_and_distance():
    assert edit_distance("ꯃꯤꯇꯩ", "ꯃꯇꯩ") == 1 and edit_distance("", "ab") == 2
    b = bootstrap(["ab", "cd", "ef", "gh"], ["ab", "cx", "ef", "gh"], n=200)
    assert b["cer"] == 0.125 and b["wer"] == 0.25 and b["cer_95ci"][0] <= 0.125 <= b["cer_95ci"][1]


def test_tune_runs_the_stages_and_resumes(data16, tools, tmp_path, monkeypatch):
    calls = []

    def fake_generate(model, ds, printer, writers, refs=4, first=None, words=None, ref_of=None, **kw):
        calls.append(kw["steps"])
        per = ds.per_writer
        return [{"writer": w, "text": ds.texts[w * per + i], "real": ds.target(w * per + i).astype(np.float32) / 255,
                 "generated": ds.target(w * per + i).astype(np.float32) / 255}
                for w in writers for i in range(first, first + words)]

    def fake_read(tar, out):
        return {"words": 1, "greedy": {"cer": 0.1, "wer": 0.2}, "with_lm": {"cer": 0.05, "wer": 0.1}}

    def fake_style(fake, real, which):      # the folder name holds the setting: prefer guidance 3, 8 references
        name = Path(fake).parent.name
        return {"hwd": 1.0 - 0.1 * ("_g3.0_" in name) - 0.2 * ("_r8_" in name) - 0.05 * ("dpmpp12" in name)}

    monkeypatch.setattr(protocol, "generate_writers", fake_generate)
    ds = WordSet(data16 / "val")
    ev = protocol.Evaluator(final_pt(tmp_path / "final.pt"), ds, tools[2], range(3), fake_read, fake_style,
                            tmp_path / "work", device="cpu")
    out = tmp_path / "grid.json"
    res = protocol.tune(ev, out, log=lambda *a: None)
    assert len(res["rows"]) == 3 + 9 + 4 + 4
    assert res["chosen"]["guidance"] == 3.0 and res["chosen"]["sampler"] == "dpmpp" and res["chosen"]["steps"] == 12
    assert res["chosen"]["refs"] == 8 and set(res["stages"]) == {"average", "guidance", "sampler", "references"}
    row = next(iter(res["rows"].values()))
    assert row["reference_greedy_cer"] == 0.1 and row["nfe"] == 35
    n = len(calls)
    res2 = protocol.tune(ev, out, log=lambda *a: None)       # resumed: nothing run again
    assert len(calls) == n and res2["chosen"] == res["chosen"]
    assert protocol.chosen_setting(out)["refs"] == 8 and protocol.chosen_setting(None) == protocol.BASE


def test_recognise_and_score_script(data16, recogniser_dir, word_repo, tmp_path):
    from mayek_diffusion.evaluate import write_fixed_set

    ds = WordSet(data16 / "test")
    write_fixed_set([ds.target(i).astype(np.float32) / 255 for i in range(5)], list(ds.texts[:5]), tmp_path / "s.tar")
    r = recognise(tmp_path / "s.tar", recogniser_dir, word_repo, tmp_path / "r.json", predictions=tmp_path / "p.tsv",
                  device="cpu", processes=1)
    assert r["words"] == 5 and 0 <= r["greedy"]["cer"] and r["with_lm"] is not None
    refs, hyps = read_predictions(tmp_path / "p.tsv")
    assert refs == list(ds.texts[:5]) and len(hyps) == 5
    weights = final_pt(tmp_path / "final.pt")
    out = subprocess.run([sys.executable, "scripts/evaluate.py", "score", "--word-repo", str(word_repo),
                          "--weights", str(weights), "--data", str(data16), "--split", "test", "--writers", "3",
                          "--recogniser", str(recogniser_dir), "--work", str(tmp_path / "work"),
                          "--out", str(tmp_path / "test.json"), "--steps", "2", "--sampler", "dpmpp", "--control",
                          "--no-style", "--sheet", str(tmp_path / "sheet.png"), "--device", "cpu"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-3000:]
    res = json.loads((tmp_path / "test.json").read_text())
    assert res["words"] == 36 and res["setting"]["steps"] == 2 and "control" in res
    assert res["recogniser"]["sha"] == "0" * 40 and (tmp_path / "sheet.png").exists()
    assert res["reference_greedy_cer"] is not None and res["seconds_per_word"] > 0
