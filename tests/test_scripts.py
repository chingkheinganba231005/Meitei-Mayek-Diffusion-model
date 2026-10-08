import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r.stdout


@pytest.mark.slow
def test_render_then_train(word_repo, tmp_path):
    from mayek_words.glyphs import GlyphStore

    glyphs, lex = tmp_path / "glyphs", tmp_path / "lexicon"
    glyphs.mkdir()
    lex.mkdir()
    store = GlyphStore.from_font()
    words = sorted({line.split("\t")[1] for line in (word_repo / "real_words" / "labels.tsv").read_text(
        encoding="utf-8").splitlines() if line})
    for split in ("train", "val", "test"):
        store.save(glyphs / f"{split}.npz")
        (lex / f"{split}.tsv").write_text("".join(f"{w}\t3\n" for w in words), encoding="utf-8")
    data = tmp_path / "data"
    run("scripts/render_data.py", "--word-repo", word_repo, "--glyphs", glyphs, "--lexicon", lex, "--out", data,
        "--writers", "train=6", "val=3", "test=3", "--words", "8", "--processes", "2")
    assert len(list((data / "train").glob("shard_*.npz"))) == 1
    out = run("scripts/train.py", "--word-repo", word_repo, "--data", data, "--run", tmp_path / "run",
              "--local", tmp_path / "local", "--steps", "4", "--device", "cpu", "--tiny", "--save-minutes", "1e9")
    assert '"done": true' in out
    assert (tmp_path / "run" / "final.pt").exists() and (tmp_path / "run" / "samples" / "final.png").exists()
    out = run("scripts/overfit_check.py", "--word-repo", word_repo, "--data", data, "--out", tmp_path / "overfit.png",
              "--steps", "2", "--min-iou", "0", "--device", "cpu")
    assert '"passed": true' in out and (tmp_path / "overfit.png").exists()
