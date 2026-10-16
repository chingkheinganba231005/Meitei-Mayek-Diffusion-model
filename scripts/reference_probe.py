"""How much a trained generator depends on which reference words it is given.

    python scripts/reference_probe.py --word-repo /content/word_repo \\
        --weights .../runs/study/coverage_seed0/final.pt --data .../data/val --out .../probe.json

On validation words, at fixed noise levels (0.3, 1 and 3) and the same noise, the denoiser is run
with two disjoint reference sets of the word's own writer that show the same letters of the word
(as the training study draws them), and with the first set of another writer. Reported per noise
level: the change between the two sets of the same writer (the mean squared difference of the
predictions over c_out^2, as the training penalty measures it), and how much worse the other
writer's references denoise the word (the difference of EDM's weighted error). A small change with
a large wrong-writer gap is a generator that follows the writer and not the particular words; a
small change with no gap would be a generator that ignores the references. The printed prior is
the word's own, so this probes the denoiser, not the generated words (scripts/evaluate.py).
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word-repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="a rendered split, e.g. data/val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-rel", type=float, default=0.10)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=912)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), args.word_repo]
    import numpy as np
    import torch

    from mayek_diffusion.data import WordSet
    from mayek_diffusion.edm import EDM
    from mayek_diffusion.reference_evidence import PairedBatcher
    from mayek_diffusion.release import load_weights
    from mayek_diffusion.train import to_device

    model, info = load_weights(args.weights, args.sigma_rel, args.device)
    torch.manual_seed(args.seed)
    batches = PairedBatcher(WordSet(args.data), pixels=8 * 256, refs=4, seed=args.seed, max_batch=8, pair_fraction=1.0)
    edm, rows = EDM(), []
    with torch.no_grad():
        for _ in range(args.batches):
            raw = batches.sample()
            b = to_device(raw, args.device)
            B = len(b["target"])
            other = torch.tensor([next((j for j in range(B) if b["writer"][j] != b["writer"][i]), i) for i in range(B)],
                                 device=b["target"].device)
            ca = model.condition(b["text"], b["ref"], b["ref_width"], b["ref_mask"])
            cb = model.condition(b["text"], b["ref_alt"], b["ref_width_alt"], b["ref_mask_alt"])
            cw = model.condition(b["text"], b["ref"][other], b["ref_width"][other], b["ref_mask"][other])
            prior = b["prior"] if info.get("prior") is not False else torch.full_like(b["prior"], -1.0)
            noise = torch.randn_like(b["target"])
            for s in (0.3, 1.0, 3.0):
                sigma = torch.full((B,), s, device=b["target"].device)
                x = b["target"] + s * noise
                den = lambda c: edm.denoise(lambda xi, cn: model(xi, prior, cn, c), x, sigma)
                da, db, dw = den(ca), den(cb), den(cw)
                sq = lambda v: v.square().mean((1, 2, 3))
                change = sq(da - db) / edm.coefficients(sigma)[1].square()
                gap = edm.weight(sigma) * (sq(dw - b["target"]) - sq(da - b["target"]))
                for i in range(B):
                    rows.append({"index": int(raw["index"][i]), "sigma": s, "matched": bool(raw["pair_weight"][i] > 0),
                                 "same_writer_change": float(change[i]),
                                 "wrong_writer_gap": float(gap[i]) if int(other[i]) != i else None})
    summary = []
    for s in (0.3, 1.0, 3.0):
        sel = [r for r in rows if r["sigma"] == s and r["matched"]]
        gaps = [r["wrong_writer_gap"] for r in sel if r["wrong_writer_gap"] is not None]
        summary.append({"sigma": s, "words": len(sel),
                        "same_writer_change": float(np.mean([r["same_writer_change"] for r in sel])) if sel else None,
                        "wrong_writer_gap": float(np.mean(gaps)) if gaps else None})
    result = {"weights": str(args.weights), "sigma_rel": args.sigma_rel, "seed": args.seed, "summary": summary,
              "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
