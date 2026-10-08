import numpy as np
import torch

from mayek_diffusion.data import Batcher, WordSet
from mayek_diffusion.model import Generator, ModelConfig
from mayek_diffusion.sample import generate
from mayek_diffusion.train import TrainConfig, to_device, train

TINY = ModelConfig(channels=(32, 64, 64), d=64, text_layers=1, style_layers=1)


def batch(data_dir, pixels=8 * 128):
    return to_device(Batcher(WordSet(data_dir / "train"), pixels=pixels, refs=4, seed=1).sample(), "cpu")


def test_forward_and_masks(data_dir):
    torch.manual_seed(0)
    m = Generator(TINY).eval()
    b = batch(data_dir)
    cond = m.condition(b["text"], b["ref"], b["ref_width"], b["ref_mask"])
    B, _, H, W = b["target"].shape
    x = torch.randn(B, 1, H, W)
    with torch.no_grad():
        y1 = m(x, b["prior"], torch.zeros(B), cond)
        assert y1.shape == (B, 1, H, W)
        # tokens behind the mask do not count
        tok = cond["tokens"].clone()
        tok[~cond["mask"]] = 100.0
        y2 = m(x, b["prior"], torch.zeros(B), {**cond, "tokens": tok})
        assert torch.allclose(y1, y2, atol=1e-4)
        # an unused reference slot does not change the style
        r = b["ref"].clone()
        r[~b["ref_mask"]] = 1.0
        t1, v1, g1 = m.style(b["ref"], b["ref_width"], b["ref_mask"])
        t2, v2, g2 = m.style(r, b["ref_width"], b["ref_mask"])
        assert torch.allclose(g1, g2, atol=1e-4)


def test_null_style_and_content(data_dir):
    m = Generator(TINY)
    b = batch(data_dir)
    B = b["text"].shape[0]
    drop = torch.zeros(B, dtype=torch.bool)
    drop[0] = True
    c = m.condition(b["text"], b["ref"], b["ref_width"], b["ref_mask"], drop_style=drop, drop_content=drop)
    n = TINY.null_tokens
    T = b["text"].shape[1]
    assert c["mask"][0, :1].all() and not c["mask"][0, 1:T].any()           # NULL only
    assert c["mask"][0, T:T + n].all() and not c["mask"][0, T + n:].any()   # null style tokens only
    assert torch.equal(c["g"][0], m.null_style) and not torch.equal(c["g"][1], m.null_style)


def test_train_resume_and_generate(data_dir, tools, tmp_path):
    cfg = TrainConfig(steps=6, warmup=2, pixels=8 * 128, log_every=2, val_every=4, save_minutes=1e9)
    run, local = tmp_path / "run", tmp_path / "local"
    r1 = train(cfg, data_dir, run, TINY, device="cpu", local_dir=local, stop_after=3)
    assert r1["step"] == 3 and not r1["done"] and (run / "last.pt").exists()
    r2 = train(cfg, data_dir, run, TINY, device="cpu", local_dir=local)
    assert r2["step"] == 6 and r2["done"] and (run / "final.pt").exists()
    assert sorted(p.name for p in run.glob("snap_*.pt")) == ["snap_0000001.pt", "snap_0000003.pt", "snap_0000004.pt"]
    ck = torch.load(run / "last.pt", weights_only=False)
    assert ck["ema"]["t"] == 6 and ck["step"] == 6
    final = torch.load(run / "final.pt", weights_only=False)
    m = Generator(TINY)
    m.load_state_dict(final["ema"][0.10])
    ds = WordSet(data_dir / "val")
    refs = [ds.target(i).astype(np.float32) / 255 for i in range(3)]
    out = generate(m, ["ꯃꯤꯇꯩ", "ꯃꯌꯦꯛ"], refs, tools[2], steps=4, seed=0)
    assert len(out) == 2 and all(o.shape[0] == 64 and 0 <= o.min() and o.max() <= 1 for o in out)
    out2 = generate(m, ["ꯃꯤꯇꯩ", "ꯃꯌꯦꯛ"], refs, tools[2], steps=4, seed=0, sampler="heun")
    assert out2[0].shape == out[0].shape
    rand = generate(m, ["ꯃꯤꯇꯩ"], None, tools[2], steps=3, seed=1)      # a random hand
    assert rand[0].shape[0] == 64


def test_steps_set_from_measured_speed(data_dir, tmp_path):
    cfg = TrainConfig(steps=0, hours=0.002, measure_skip=1, measure_steps=2, warmup=2, pixels=8 * 128,
                      log_every=1000, val_every=1000, save_minutes=1e9, snapshots=())
    r = train(cfg, data_dir, tmp_path / "run", TINY, device="cpu", local_dir=tmp_path / "local")
    assert r["total"] > 0 and r["done"] and r["step"] >= r["total"] - 0


def test_default_model_has_64_wide_attention_heads():
    """diffusers' attention_head_dim is the number of heads: the default model must still get heads
    64 wide, and its size must stay as planned."""
    m = Generator(ModelConfig())
    blocks = [t for mod in m.denoiser.unet.modules() if hasattr(mod, "transformer_blocks")
              for t in mod.transformer_blocks]
    assert blocks
    for t in blocks:
        for attn in (t.attn1, t.attn2):
            assert attn.inner_dim // attn.heads == 64
    n = sum(p.numel() for p in m.parameters()) / 1e6
    assert 110 < n < 117, n


def test_references_are_a_set(data_dir, tools):
    """The order of the references does not matter, and more of them than in training (8) work."""
    torch.manual_seed(0)
    m = Generator(TINY).eval()
    b = batch(data_dir)
    keep = b["ref_mask"].all(1)
    ref, rw, rm = b["ref"][keep], b["ref_width"][keep], b["ref_mask"][keep]
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        g1 = m.style(ref, rw, rm)[2]
        g2 = m.style(ref[:, perm], rw[:, perm], rm[:, perm])[2]
    assert torch.allclose(g1, g2, atol=1e-5)
    ds = WordSet(data_dir / "val")
    refs = [ds.target(i).astype(np.float32) / 255 for i in range(8)]
    out = generate(m, ["ꯃꯤꯇꯩ"], refs, tools[2], steps=3)
    assert out[0].shape[0] == 64
