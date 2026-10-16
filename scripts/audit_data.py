"""Checks of the rendered splits that read only their labels and metadata (no images).

    python scripts/audit_data.py --word-repo /content/word_repo \\
        --data /content/drive/MyDrive/meitei-word-diffusion/data --out .../results/data_audit.json

Per split: the words, writers and distinct texts; texts the generator cannot take (characters
outside its alphabet, longer than 48 characters); writers with the same text twice among their 16
words; canvas widths; the spread of the handwriting's width relative to print. Across splits: the
texts that two splits share (numbers and composed words can repeat even though the word lists
are disjoint). For the training study: how often two disjoint reference sets showing the same
letters of the target can be found (reference_evidence.match_sets), and the share of the target's
letters they show, on a sample of words of each split.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word-repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=5000, help="words per split for the pair statistics")
    args = ap.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), args.word_repo]
    import numpy as np

    from mayek_diffusion.data import CHAR_ID
    from mayek_diffusion.reference_evidence import match_sets

    out, vocab = {"splits": {}}, {}
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        root = Path(args.data) / split
        if not (root / "info.json").exists():
            continue
        info = json.loads((root / "info.json").read_text(encoding="utf-8"))
        per = info["words_per_writer"]
        texts, widths, ratios, repeats, writers = [], [], [], 0, 0
        for name in info["shards"]:
            with np.load(root / name, allow_pickle=False) as z:
                t, offsets = [str(x) for x in z["texts"]], z["offsets"]
                if len(t) % per or len(offsets) != len(t) + 1 or (np.diff(offsets) <= 0).any():
                    raise SystemExit(f"{root / name}: inconsistent shard")
                if not (z["writer"].reshape(-1, per) == z["writer"].reshape(-1, per)[:, :1]).all():
                    raise SystemExit(f"{root / name}: a writer's words are not together")
                for k in range(0, len(t), per):
                    writers += 1
                    repeats += len(set(t[k:k + per])) < per
                texts += t
                widths += np.diff(offsets).tolist()
                ratios += np.log((z["ink"][:, 1] - z["ink"][:, 0]) / z["print_width"]).tolist()
        vocab[split] = set(texts)
        found, share = [], []
        for i in rng.choice(len(texts), min(args.sample, len(texts)), replace=False):
            first = (i // per) * per
            others = sorted({texts[j] for j in range(first, first + per) if texts[j] != texts[i]})
            k = 4 if rng.random() < 0.5 else int(rng.integers(1, 5))
            m = match_sets(texts[i], others, k, rng)
            found.append(bool(m and m[2] > 0))
            share.append(m[2] if m else 0.0)
        out["splits"][split] = {
            "words": len(texts), "writers": writers, "distinct_texts": len(vocab[split]),
            "texts_outside_alphabet": sum(any(c not in CHAR_ID for c in t) for t in texts),
            "texts_over_48": sum(len(t) > 48 for t in texts), "empty_texts": sum(not t for t in texts),
            "writers_with_a_repeated_text": repeats,
            "canvas_width": {q: float(np.percentile(widths, p)) for q, p in (("min", 0), ("p50", 50), ("p99", 99),
                                                                              ("max", 100))},
            "log_width_ratio_sd": round(float(np.std(ratios)), 4),
            "matched_pairs": round(float(np.mean(found)), 4),
            "matched_pair_mean_coverage": round(float(np.mean([s for s in share if s > 0] or [0])), 4),
            "mean_pair_weight": round(float(np.mean(share)), 4)}
        print(split, json.dumps(out["splits"][split]), flush=True)
    names = list(vocab)
    out["shared_texts"] = {f"{a}_{b}": len(vocab[a] & vocab[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    print("shared texts:", out["shared_texts"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
