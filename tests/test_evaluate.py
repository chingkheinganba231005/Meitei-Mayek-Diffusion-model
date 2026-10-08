import numpy as np
import torch

from mayek_diffusion.data import WordSet
from mayek_diffusion.evaluate import generate_writers, write_fixed_set, write_writer_folders
from mayek_diffusion.model import Generator, ModelConfig
from mayek_diffusion.sample import sheet

TINY = ModelConfig(channels=(32, 64, 64), d=64, text_layers=1, style_layers=1)


def test_sets_read_back(data_dir, tools, tmp_path):
    from mayek_htr.data import FixedSet

    torch.manual_seed(0)
    m = Generator(TINY)
    ds = WordSet(data_dir / "val")
    rows = generate_writers(m, ds, tools[2], writers=[0, 1], refs=4, steps=3)
    assert len(rows) == 2 * 4 and all(r["generated"].shape[0] == 64 for r in rows)
    write_fixed_set([r["real"] for r in rows], [r["text"] for r in rows], tmp_path / "real.tar")
    fs = FixedSet(tmp_path / "real.tar")
    assert fs.texts == [r["text"] for r in rows] and all(im.shape[0] == 64 for im in fs.images)
    write_writer_folders([r["generated"] for r in rows], [r["writer"] for r in rows], tmp_path / "hwd")
    assert len(list((tmp_path / "hwd").rglob("*.png"))) == 8
    sheet(m, ds, tools[2], tmp_path / "sheet.png", writers=2, steps=3)
    assert (tmp_path / "sheet.png").exists()
