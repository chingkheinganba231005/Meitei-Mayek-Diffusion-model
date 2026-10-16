from dataclasses import replace

import numpy as np
import pytest
import torch

from mayek_diffusion.data import WordSet
from mayek_diffusion.edm import EDM, LogVar
from mayek_diffusion.model import Generator, ModelConfig
from mayek_diffusion.reference_evidence import PairedBatcher, consistency, coverage, match_sets
from mayek_diffusion.train import Copier, TrainConfig, step_losses, to_device, train

TINY = ModelConfig(channels=(32, 64, 64), d=64, text_layers=1, style_layers=1)
TEXTS = ["ꯃꯤꯇꯩ", "ꯃꯌꯦꯛ", "ꯑꯃ", "ꯇꯦ", "ꯃꯔꯨ", "ꯊꯥꯗ", "ꯄꯥꯡꯏ", "ꯇꯥꯀꯄꯒꯦ"]


def test_matched_sets_show_the_same_letters_of_the_target():
    rng = np.random.default_rng(0)
    for k in (1, 2, 3, 4):
        for _ in range(20):
            a, b, w = match_sets("ꯃꯇꯦ", TEXTS, k, rng)
            assert len(a) == len(b) and not set(a) & set(b)
            if w > 0:
                ca, cb = coverage("ꯃꯇꯦ", [TEXTS[i] for i in a]), coverage("ꯃꯇꯦ", [TEXTS[i] for i in b])
                assert ca == cb and w == len(ca) / 3
    a, b, w = match_sets("ꯃꯇꯦ", TEXTS, 4, rng, matched=False)
    assert w == 1.0 and len(a) == len(b) and not set(a) & set(b)
    assert match_sets("ꯃꯇꯦ", ["ꯑꯃ"], 2, rng) is None
    a, b, w = match_sets("ꯐ", ["ꯑꯃ", "ꯇꯦ"], 1, rng)       # nothing to match on: an unmatched pair
    assert w == 0.0 and len(a) == len(b) == 1


def test_pairs_exclude_the_target_text_and_each_other(data16):
    ds = WordSet(data16 / "val")
    b = PairedBatcher(ds, pixels=8 * 128, refs=4, seed=5, pair_fraction=0.5).sample()
    assert b["pair_valid"].any() and (b["pair_weight"] > 0).any()
    assert b["second"].sum() == round(0.5 * len(b["second"]))
    for row, i in enumerate(b["index"]):
        if not b["pair_valid"][row]:
            continue
        a = b["ref_index"][row][b["ref_mask"][row]]
        c = b["ref_index_alt"][row][b["ref_mask_alt"][row]]
        assert len(a) == len(c) and not set(a) & set(c)
        assert ds.texts[i] not in ds.texts[a] and ds.texts[i] not in ds.texts[c]
        assert not set(ds.texts[a]) & set(ds.texts[c])
        assert all(ds.writer[j] == ds.writer[i] for j in [*a, *c])
        if b["pair_weight"][row] > 0:
            assert coverage(ds.texts[i], ds.texts[a]) == coverage(ds.texts[i], ds.texts[c])


def test_penalty_scale_and_gradients():
    edm = EDM()
    s = torch.tensor([1.0, 1.0, 0.01])
    a = torch.ones(3, 1, 2, 2, requires_grad=True)
    b = torch.zeros_like(a, requires_grad=True)
    per = consistency(a, b, s, edm, torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(per[0], 1 / edm.coefficients(s)[1][0].square()) and per[1:].abs().sum() == 0
    per.sum().backward()
    assert a.grad[0].abs().sum() > 0 and b.grad[0].abs().sum() > 0 and a.grad[1:].abs().sum() == 0


def batch(data16, n=4, seed=2):
    raw = PairedBatcher(WordSet(data16 / "val"), pixels=8 * 128, refs=2, seed=seed, pair_fraction=1.0).sample()
    return to_device({k: v[:n] for k, v in raw.items()}, "cpu")


def test_variants(data16):
    torch.manual_seed(4)
    b = batch(data16)
    m, edm, lv = Generator(TINY), EDM(), LogVar()
    sigma, noise = torch.ones(4), torch.randn_like(b["target"])
    cfg = TrainConfig(reference_objective="paired")
    run = lambda c: step_losses(m, lv, edm, b, c, drop=False, sigma=sigma, noise=noise)
    l_paired, _ = run(cfg)
    l_zero, _ = run(replace(cfg, reference_objective="coverage", w_reference=0.0))
    assert torch.allclose(l_paired, l_zero)              # the penalty is the only difference
    l_single, t_single = run(replace(cfg, reference_objective="single"))
    assert t_single["penalty_mass"] == 0
    b["pair_weight"].fill_(1.0)
    b["pair_valid"].fill_(True)
    l_cov, terms = run(replace(cfg, reference_objective="coverage"))
    assert terms["penalty_mass"] == 1 and l_cov >= l_paired
    l_cov.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    for objective in ("ungated", "random"):
        loss, terms = run(replace(cfg, reference_objective=objective))
        assert torch.isfinite(loss) and 0 <= float(terms["penalty_mass"]) <= 1


def test_seed_sets_initial_weights_and_resume_needs_the_same_settings(data_dir, tmp_path):
    cfg = TrainConfig(steps=2, pixels=8 * 128, seed=13, snapshots=())
    states = []
    for name in ("a", "b"):
        train(cfg, data_dir, tmp_path / name, TINY, device="cpu", local_dir=tmp_path / (name + "_local"), stop_after=0)
        states.append(torch.load(tmp_path / name / "last.pt", weights_only=False)["model"])
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    with pytest.raises(ValueError, match="other settings"):
        train(replace(cfg, reference_objective="coverage"), data_dir, tmp_path / "a", TINY, device="cpu",
              local_dir=tmp_path / "c")
    train(replace(cfg, save_minutes=5.0), data_dir, tmp_path / "a", TINY, device="cpu", local_dir=tmp_path / "d",
          stop_after=0)                                  # an operational setting may change


def test_copier_carries_on_after_a_failure_and_stops_after_three(tmp_path):
    c = Copier()
    for k in range(2):
        c.copy(tmp_path / "missing.pt", tmp_path / "out.pt", tries=1)
        c.wait()
    (tmp_path / "x.pt").write_bytes(b"1")
    c.copy(tmp_path / "x.pt", tmp_path / "out.pt", tries=1)
    c.wait()
    assert c.failures == 0 and (tmp_path / "out.pt").exists()
    for k in range(2):
        c.copy(tmp_path / "missing.pt", tmp_path / "out.pt", tries=1)
        c.wait()
    c.copy(tmp_path / "missing.pt", tmp_path / "out.pt", tries=1)
    with pytest.raises(RuntimeError, match="no longer reach"):
        c.wait()


def test_evaluation_leaves_out_targets_written_in_the_references(data16, tmp_path):
    from mayek_diffusion.protocol import Evaluator

    ds = WordSet(data16 / "val")
    ds.texts[4] = ds.texts[0]          # a target with the text of its own reference
    ds.texts[5] = ds.texts[16]         # ... and of the next writer's (the control's)
    ev = Evaluator(None, ds, None, [0, 1, 2], None, None, tmp_path, device="cpu")
    chosen = ev.target_indices(4, 12, 4)
    assert 4 not in chosen and 5 not in chosen and 6 in chosen
    assert 4 in Evaluator(None, ds, None, [0, 1, 2], None, None, tmp_path, device="cpu",
                          strict=False).target_indices(4, 12, 4)


def test_a_study_run_trains_and_loads(data_dir, tmp_path):
    from mayek_diffusion.release import load_weights

    cfg = TrainConfig(steps=2, pixels=8 * 128, refs=2, seed=3, snapshots=(), reference_objective="coverage",
                      log_every=1, val_every=10)
    assert train(cfg, data_dir, tmp_path / "run", TINY, device="cpu", local_dir=tmp_path / "local")["done"]
    m, info = load_weights(tmp_path / "run" / "final.pt", 0.10, "cpu")
    assert all(torch.isfinite(p).all() for p in m.parameters())
    ck = torch.load(tmp_path / "run" / "final.pt", weights_only=False)
    assert ck["cfg"]["reference_objective"] == "coverage"
    assert {"consistency", "penalty_mass", "matched"} <= set(ck["history"][0])
