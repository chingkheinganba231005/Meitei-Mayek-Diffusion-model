import json

import numpy as np
import torch
from PIL import Image

from mayek_diffusion.canvas import CANVAS_H, from_real
from mayek_diffusion.data import Batcher, WordSet
from mayek_diffusion.finetune import AdaptConfig, adapt, checkpoint
from mayek_diffusion.model import Generator, ModelConfig, config_dict
from mayek_diffusion.owner import prepare, references, split, write_writer
from mayek_diffusion.release import export, load, load_weights
from mayek_diffusion.sample import generate

TINY = ModelConfig(channels=(32, 64, 64), d=64, text_layers=1, style_layers=1)


def test_owner_split_and_sets(word_repo, tools, tmp_path):
    s = split(100)
    perm = np.random.default_rng(2026).permutation(100)
    assert s["adapt"] == perm[:70].tolist() and s["held_out"] == perm[70:].tolist()
    assert s["references"] == s["adapt"][:4]
    real = word_repo / "real_words"
    out = prepare(real, tmp_path / "owner", tools[2])
    assert out["adapt"] == s["adapt"] and len(out["files"]["held_out"]) == 30 and out["ink_lost_max"] < 0.02
    adapt_set, held = WordSet(tmp_path / "owner" / "adapt"), WordSet(tmp_path / "owner" / "held_out")
    assert len(adapt_set) == adapt_set.per_writer == 70 and len(held) == 30
    assert list(adapt_set.texts[:4]) == out["texts"]["references"]
    refs = references(tmp_path / "owner")
    first = from_real(np.asarray(Image.open(real / "images" / out["files"]["references"][0]).convert("L")))[0]
    assert len(refs) == 4 and refs[0].shape == first.shape and np.abs(refs[0] - first).max() <= 1 / 255 + 1e-6
    b = Batcher(adapt_set, pixels=8 * 256, refs=4, seed=0).sample()
    assert b["ref"].shape[1] == 4 and (b["writer"] == 0).all() and np.isfinite(b["log_width_ratio"]).all()


def test_release_round_trip(tmp_path, tools):
    torch.manual_seed(0)
    m = Generator(TINY).eval()
    export(m, tmp_path / "rel", sampling={"guidance": 1.5}, source={"run": "test", "step": 6})
    loaded, config = load(tmp_path / "rel")
    assert config["sampling"]["guidance"] == 1.5 and config["sampling"]["steps"] == 18
    assert config["canvas"]["baseline_row"] == 46 and len(config["alphabet"]) == 54
    rounded = Generator(TINY).eval()
    rounded.load_state_dict({k: v.half().float() for k, v in m.state_dict().items()})
    refs = [np.zeros((CANVAS_H, 40), np.float32)]
    refs[0][30:46, 5:35] = 1.0
    a = generate(loaded, ["ꯃꯤꯇꯩ", "ꯃꯌꯦꯛ"], refs, tools[2], steps=3, seed=4)
    b = generate(rounded, ["ꯃꯤꯇꯩ", "ꯃꯌꯦꯛ"], refs, tools[2], steps=3, seed=4)
    assert all(np.allclose(x, y, atol=1e-5) for x, y in zip(a, b))
    c = generate(m, ["ꯃꯤꯇꯩ"], refs, tools[2], steps=3, seed=4)
    assert np.abs(c[0] - a[0]).max() < 0.05          # float16 weights change little
    # an adapted generator shares config.json
    export(m, tmp_path / "rel", name="owner_generator")
    assert load(tmp_path / "rel", "owner_generator")[1]["sampling"]["guidance"] == 1.5
    # a run's files
    sd = m.state_dict()
    torch.save({"ema": {0.05: sd, 0.1: sd, 0.15: sd}, "model_cfg": config_dict(TINY), "cfg": {"prior": False},
                "step": 6}, tmp_path / "final.pt")
    torch.save({"ema": {k: v.bfloat16() for k, v in sd.items()}, "model_cfg": config_dict(TINY), "step": 3,
                "prior": True}, tmp_path / "snap.pt")
    m1, info = load_weights(tmp_path / "final.pt", 0.15)
    assert info["kind"] == "final" and info["prior"] is False and info["sigma_rel"] == 0.15
    m2, info = load_weights(tmp_path / "snap.pt")
    assert info["kind"] == "snapshot" and info["prior"] is True
    assert torch.equal(next(m1.parameters()), next(m.parameters()))
    m3, info = load_weights(tmp_path / "rel")
    assert info["kind"] == "release" and json.loads((tmp_path / "rel" / "config.json").read_text())["prior"]


def test_adapt(data_dir, tmp_path, tools):
    val = WordSet(data_dir / "val")
    items = range(val.per_writer)                      # writer 0 as "the owner"
    write_writer([val.target(i).astype(np.float32) / 255 for i in items], [val.texts[i] for i in items],
                 tmp_path / "owner" / "adapt", tools[2])
    owner = WordSet(tmp_path / "owner" / "adapt")
    torch.manual_seed(0)
    m = Generator(TINY)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    cfg = AdaptConfig(steps=4, warmup=1, owner_pixels=8 * 128, synth_pixels=8 * 128, log_every=2, decay=0.5)
    avg, history = adapt(m, owner, WordSet(data_dir / "train"), cfg, device="cpu")
    assert len(history) == 2 and "owner_loss" in history[0] and "synthetic_loss" in history[0]
    assert all(np.isfinite(v) for row in history for v in row.values())
    assert any(not torch.equal(before[k], v) for k, v in avg.state_dict().items())
    torch.save(checkpoint(avg, cfg, history, {"sigma_rel": 0.10, "from": "final.pt"}), tmp_path / "owner.pt")
    m2, info = load_weights(tmp_path / "owner.pt")
    assert info["kind"] == "adapted" and info["sigma_rel"] == 0.10 and info["prior"] is True
    out = generate(m2, ["ꯃꯤꯇꯩ"], references(tmp_path / "owner", 2), tools[2], steps=3)
    assert out[0].shape[0] == CANVAS_H
