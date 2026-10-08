import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mayek_diffusion.model import Generator, ModelConfig, config_dict
from mayek_diffusion.protocol import BASE

ROOT = Path(__file__).resolve().parents[1]
TINY = ModelConfig(channels=(32, 64, 64), d=64, text_layers=1, style_layers=1)


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r.stdout


@pytest.mark.slow
def test_owner_prepare_adapt_score(word_repo, data_dir, data16, recogniser_dir, tmp_path):
    torch.manual_seed(0)
    sd = Generator(TINY).state_dict()
    torch.save({"ema": {0.05: sd, 0.1: sd, 0.15: sd}, "model_cfg": config_dict(TINY), "cfg": {"prior": True},
                "step": 6}, tmp_path / "final.pt")
    (tmp_path / "grid.json").write_text(json.dumps({"chosen": dict(BASE, sampler="dpmpp", steps=2)}))
    owner, results = tmp_path / "owner", tmp_path / "results"
    out = run("scripts/owner.py", "prepare", "--word-repo", word_repo, "--real", word_repo / "real_words",
              "--out", owner, "--split", results / "owner_split.json")
    assert '"adapt": 70' in out and (owner / "adapt" / "info.json").exists()
    split = json.loads((results / "owner_split.json").read_text())
    assert len(split["files"]["references"]) == 4 and len(split["held_out"]) == 30
    out = run("scripts/owner.py", "adapt", "--word-repo", word_repo, "--weights", tmp_path / "final.pt",
              "--setting", tmp_path / "grid.json", "--owner", owner, "--data", data_dir, "--train-shards", 1,
              "--out", tmp_path / "owner.pt", "--steps", 2, "--tiny", "--device", "cpu")
    assert (tmp_path / "owner.pt").exists()
    out = run("scripts/owner.py", "score", "--word-repo", word_repo, "--weights", tmp_path / "final.pt",
              "--adapted", tmp_path / "owner.pt", "--setting", tmp_path / "grid.json", "--owner", owner,
              "--data", data16, "--recogniser", recogniser_dir, "--work", tmp_path / "work",
              "--out", results / "owner.json", "--sheet", tmp_path / "owner.png", "--seeds", 1,
              "--val-writers", 2, "--no-style", "--device", "cpu")
    res = json.loads((results / "owner.json").read_text())
    assert res["held_out"] == 30 and res["few_shot"]["recognised"]["words"] == 30
    assert res["adapted"]["validation_writers"]["writers"] == 2 and (tmp_path / "owner.png").exists()
