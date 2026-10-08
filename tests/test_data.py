import numpy as np
from PIL import Image

from mayek_diffusion.canvas import BASE_ROW, CANVAS_H, L_PX, from_real, ink_columns, stretch
from mayek_diffusion.data import NULL_ID, NUM_CHARS, Batcher, WordSet, encode
from mayek_diffusion.writers import ON_LINE, from_sample


def test_writer_keeps_its_style(tools):
    synth, words, _ = tools
    rng = np.random.default_rng(3)
    styles = []
    for _ in range(2):
        w = synth.new_writer(rng)
        ss = [synth.render(words.text(rng), rng).style for _ in range(5)]
        for s in ss:
            assert abs(np.log(s["L"] / w["L"])) < 0.2 and s["marks_beside"] == w["marks_beside"]
            assert s["touch"] == w["touch"] and s["uneven"] == w["uneven"] and s["blur"] == w["blur"]
        styles.append(w)
    assert styles[0]["L"] != styles[1]["L"]


def test_canvas_puts_the_baseline_on_its_row(tools):
    synth, words, _ = tools
    rng = np.random.default_rng(5)
    synth.new_writer(rng)
    for _ in range(20):
        s = synth.render(words.text(rng), rng)
        x, lost = from_sample(s)
        assert x.shape[0] == CANVAS_H and lost < 0.02
        scale = L_PX / s.style["L"]
        bottoms = [y1 * scale for ch, x0, y0, x1, y1 in s.boxes if ch in ON_LINE]
        if bottoms:   # the median bottom of the letters lands on BASE_ROW (rounding: a pixel)
            shift = BASE_ROW - np.median([y1 for ch, x0, y0, x1, y1 in s.boxes if ch in ON_LINE]) * scale
            assert abs(np.median(bottoms) + shift - BASE_ROW) < 1.0


def test_printer_and_stretch(tools):
    _, _, printer = tools
    p = printer("ꯃꯤꯇꯩ")
    assert np.array_equal(p, printer("ꯃꯤꯇꯩ"))
    out = stretch(p, 10, 90, 128)
    c = ink_columns(out)
    assert out.shape == (CANVAS_H, 128) and abs(c[0] - 10) <= 1 and abs(c[1] - 90) <= 1


def test_real_words_fit_the_canvas(word_repo):
    names = sorted((word_repo / "real_words" / "images").glob("*.png"))[:30]
    for n in names:
        x, lost = from_real(np.asarray(Image.open(n).convert("L")))
        assert x.shape[0] == CANVAS_H and lost < 0.02 and x.shape[1] <= 384


def test_shards_are_consistent(data_dir):
    ds = WordSet(data_dir / "train")
    assert len(ds) == 48 and ds.per_writer == 8
    for i in range(len(ds)):
        t, p = ds.target(i), ds.prior(i)
        assert t.shape == p.shape and t.dtype == np.uint8
        ct, cp = ink_columns(t), ink_columns(p)
        assert tuple(ds.ink[i]) == ct and abs(cp[0] - ct[0]) <= 1 and abs(cp[1] - ct[1]) <= 1
        assert all(1 <= k <= NUM_CHARS for k in encode(ds.texts[i]))
        assert ds.writer[i] == i // 8


def test_batches(data_dir):
    ds = WordSet(data_dir / "train")
    bt = Batcher(ds, pixels=16 * 256, refs=4, seed=0)
    for _ in range(20):
        b = bt.sample()
        B, H, W = b["target"].shape
        assert H == CANVAS_H and W % 32 == 0 and b["prior"].shape == (B, H, W)
        assert b["ref"].shape[:3] == (B, 4, H) and b["ref"].shape[3] % 16 == 0
        assert (b["ref_mask"].sum(1) >= 1).all() and (b["text"][:, 0] > 0).all()
        assert (b["text"] != NULL_ID).all()
        for k, i in enumerate(b["index"]):
            assert ds.widths[i] <= W and b["width"][k] == ds.widths[i]
            assert ds.writer[i] == b["writer"][k]
        assert np.isfinite(b["log_width_ratio"]).all()
