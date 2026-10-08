import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

WORD_REPO = Path(os.environ.get("WORD_REPO", Path(__file__).resolve().parents[2] / "word_repo"))
sys.path.insert(0, str(WORD_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _split(folder, synth, words, printer, writers, n_words, seed):
    from mayek_diffusion.render import render_writers
    folder.mkdir(parents=True, exist_ok=True)
    arrays = render_writers(range(writers), n_words, seed, synth=synth, words=words, printer=printer)
    np.savez_compressed(folder / "shard_0000.npz", **arrays)
    (folder / "info.json").write_text(json.dumps({"words_per_writer": n_words, "shards": ["shard_0000.npz"]}))
    return arrays


@pytest.fixture(scope="session")
def word_repo():
    return WORD_REPO


@pytest.fixture(scope="session")
def tools(tmp_path_factory):
    """A writer synthesiser over the font's characters (no TUMMHCD needed), a lexicon made of
    the released real words, the printer."""
    from mayek_words.glyphs import GlyphStore
    from mayek_words.lexicon import Lexicon
    from mayek_words.synth import Words, load_priors

    from mayek_diffusion.writers import Printer, WriterSynth, writer_config

    lex = tmp_path_factory.mktemp("lex") / "lexicon.tsv"
    rows = [line.split("\t")[1] for line in (WORD_REPO / "real_words" / "labels.tsv").read_text(
        encoding="utf-8").splitlines() if line]
    lex.write_text("".join(f"{w}\t{5}\n" for w in sorted(set(rows))), encoding="utf-8")
    priors = load_priors()
    synth = WriterSynth(GlyphStore.from_font(), priors, writer_config())
    words = Words(synth, Lexicon.load(lex), seed=0, scrambled=0.1)
    return synth, words, Printer(priors)


@pytest.fixture(scope="session")
def data_dir(tools, tmp_path_factory):
    synth, words, printer = tools
    d = tmp_path_factory.mktemp("data")
    _split(d / "train", synth, words, printer, writers=6, n_words=8, seed=100)
    _split(d / "val", synth, words, printer, writers=3, n_words=8, seed=101)
    return d


@pytest.fixture(scope="session")
def data16(tools, tmp_path_factory):
    """Splits of 16 words a writer, as rendered for the generator (val and test: 3 writers)."""
    synth, words, printer = tools
    d = tmp_path_factory.mktemp("data16")
    _split(d / "val", synth, words, printer, writers=3, n_words=16, seed=101)
    _split(d / "test", synth, words, printer, writers=3, n_words=16, seed=102)
    return d


@pytest.fixture(scope="session")
def recogniser_dir(tmp_path_factory):
    """A recogniser folder as on Hugging Face (recogniser.pt, char_lm.pkl, decoding.json), with a
    small untrained network: the scoring plumbing runs end to end, the scores mean nothing."""
    from dataclasses import asdict

    import torch
    from mayek_htr.lm import CharLM
    from mayek_htr.model import build
    from mayek_htr.train import TrainConfig

    d = tmp_path_factory.mktemp("recogniser")
    torch.manual_seed(0)
    cfg = TrainConfig(encoder="small_cnn", init="none")
    model = build(cfg.encoder, init="none", hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout)
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "step": 0}, d / "recogniser.pt")
    words = [line.split("\t")[1] for line in (WORD_REPO / "real_words" / "labels.tsv").read_text(
        encoding="utf-8").splitlines() if line]
    lm = CharLM(order=3)
    lm.fit({w: 1.0 for w in words})
    lm.save(d / "char_lm.pkl")
    (d / "decoding.json").write_text(json.dumps({"alpha": 0.5, "beta": 1.0, "beam": 4}))
    (d / "source.json").write_text(json.dumps({"repo": "test", "sha": "0" * 40}))
    return d
