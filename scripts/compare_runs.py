"""Compare models scored on the same writers and words, with intervals over writers.

    python scripts/compare_runs.py --base .../results/study/paired_seed0.json \\
        --runs .../results/study/coverage_seed0.json .../results/study/ungated_seed0.json \\
        --out .../results/study/compare_seed0.json

Each file is the output of `scripts/evaluate.py score` (with its per-writer counts). For every run
against the base: the difference (run - base) of the greedy and language-model CER and WER,
pooled over words, and of the mean per-writer HWD, each with a 95% interval from resampling
writers (5,000 draws). The runs must have generated the same target words. These intervals cover
the choice of writers for one trained model each; the spread over training seeds is a separate
matter and is reported from runs with different seeds.
"""

import argparse
import json
import sys
from pathlib import Path

METRICS = ("cer_greedy", "cer_with_lm", "wer_greedy", "wer_with_lm", "hwd")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mayek_diffusion.scores import paired_bootstrap

    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    base = load(args.base)
    out = {"base": args.base, "comparisons": []}
    for path in args.runs:
        run = load(path)
        if run.get("targets") != base.get("targets"):
            raise SystemExit(f"{path} generated other words than {args.base}")
        row = {"run": path, "label": run.get("label")}
        for m in METRICS:
            if m == "hwd":
                a = {w: v["hwd"] for w, v in base["per_writer"].items() if "hwd" in v}
                b = {w: v["hwd"] for w, v in run["per_writer"].items() if "hwd" in v}
                if not a or not b:
                    continue
            else:
                a, b = base["per_writer"], run["per_writer"]
            row[m] = paired_bootstrap(a, b, m)
        out["comparisons"].append(row)
        print(f"{run.get('label') or path}: " + ", ".join(
            f"{m} {row[m]['b_minus_a']:+.4f} [{row[m]['ci95'][0]:+.4f}, {row[m]['ci95'][1]:+.4f}]"
            for m in METRICS if m in row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
