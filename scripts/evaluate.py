"""Choose the sampling setting on validation writers, and score generated words.

    # E3: the grid on 200 validation writers (resumable: rows already in --out are kept)
    python scripts/evaluate.py tune --word-repo /content/word_repo \\
        --weights /content/drive/MyDrive/meitei-word-diffusion/runs/main/final.pt \\
        --data /content/drive/MyDrive/meitei-word-diffusion/data --recogniser /content/recogniser \\
        --work /content/eval/val --out /content/drive/MyDrive/meitei-word-diffusion/results/eval_val_grid.json

    # E1-E2: all 1,000 test writers, once, with the setting chosen on validation
    python scripts/evaluate.py score --word-repo /content/word_repo --weights .../runs/main/final.pt \\
        --data .../data --split test --writers 1000 --setting .../results/eval_val_grid.json \\
        --recogniser /content/recogniser --control --fid-kid --work /content/eval/test \\
        --out .../results/eval_test.json --sheet .../eval/test_writers.png

tune: 200 validation writers, 12 generated words each (the texts of their words 5-16, with
words 1-4 as references), in stages: the average of the weights (sigma_rel 0.05, 0.10, 0.15);
guidance 1, 1.5, 2, 3 with the interval of noise levels where it applies (all, (0.3, 5],
(0.1, 2]); Heun with 18 steps against DPM-Solver++(2M) with 8, 12, 16 and 24; then 1, 2, 4 or
8 references (generating words 9-16). Each stage keeps the lowest HWD among the settings whose
greedy CER is at most the reference CER (the recogniser on the real words) plus 1 point.

score: one setting (the grid's choice, or --sigma-rel and the other flags) on the writers of a
split: CER and WER greedy and with the language model, the reference level, HWD, with
--fid-kid FID and KID, with --control the same texts written with the next writer's
references (HWD must then be clearly worse). --no-prior for a model trained without the
printed prior (the ablation). Snapshots and the ablation's checkpoint are scored the same way.

The recogniser folder holds recogniser.pt, char_lm.pkl and decoding.json (Hugging Face), and
optionally source.json (its repository and commit), which is copied into the results.
--no-style skips HWD, FID and KID (the HWD package missing); the rule then falls back to CER.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["tune", "score"])
    ap.add_argument("--word-repo", required=True)
    ap.add_argument("--weights", required=True, help="final.pt, snap_*.pt, last.pt, owner.pt or a release folder")
    ap.add_argument("--data", required=True, help="folder with the rendered splits")
    ap.add_argument("--recogniser", required=True, help="folder with recogniser.pt, char_lm.pkl, decoding.json")
    ap.add_argument("--work", required=True, help="local folder for the images")
    ap.add_argument("--out", required=True, help="results .json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--writers", type=int, default=200)
    ap.add_argument("--first-writer", type=int, default=0)
    ap.add_argument("--setting", help="grid results .json (its chosen setting) for score")
    ap.add_argument("--sigma-rel", type=float)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--interval", type=float, nargs=2)
    ap.add_argument("--sampler", choices=["heun", "dpmpp"])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--refs", type=int)
    ap.add_argument("--control", action="store_true", help="also the wrong-style control")
    ap.add_argument("--fid-kid", action="store_true")
    ap.add_argument("--no-prior", action="store_true", help="the model was trained without the printed prior")
    ap.add_argument("--no-style", action="store_true", help="no HWD, FID or KID")
    ap.add_argument("--sheet", help="score: a sheet of writers (references | real | generated), .png")
    ap.add_argument("--label", default="", help="score: a name stored in the results")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sys.path.insert(0, args.word_repo)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from mayek_diffusion import protocol
    from mayek_diffusion.data import WordSet
    from mayek_diffusion.scores import StyleScorer, recognise
    from mayek_diffusion.writers import Printer
    from mayek_words.synth import load_priors

    rec = Path(args.recogniser)
    source = json.loads((rec / "source.json").read_text(encoding="utf-8")) if (rec / "source.json").exists() else None
    style = None
    if not args.no_style:
        scorer = StyleScorer()
        style = lambda fake, real, which: scorer(fake, real, which)
    device = None if args.device == "cpu" else args.device
    read = lambda tar, out: recognise(tar, rec, args.word_repo, out, device=device)
    ds = WordSet(Path(args.data) / args.split)
    writers = range(args.first_writer, args.first_writer + args.writers)
    ev = protocol.Evaluator(args.weights, ds, Printer(load_priors()), writers, read, style, args.work,
                            device=args.device, batch=args.batch, prior=False if args.no_prior else None)
    extra = {"weights": str(args.weights), "split": args.split, "first_writer": args.first_writer,
             "recogniser": source, "style_scores": not args.no_style}
    if args.command == "tune":
        res = protocol.tune(ev, args.out, extra)
        print(json.dumps({"chosen": res["chosen"]}), flush=True)
        return
    setting = protocol.chosen_setting(args.setting)
    for k in ("sigma_rel", "guidance", "interval", "sampler", "steps", "refs"):
        if getattr(args, k) is not None:
            setting[k] = list(getattr(args, k)) if k == "interval" else getattr(args, k)
    first = max(4, setting["refs"])
    row = ev.run(setting, first, ds.per_writer - first, control=args.control, fid_kid=args.fid_kid)
    row.update(extra, label=args.label, setting_from=args.setting)
    if args.sheet:
        protocol.writer_sheet(ev, ev.last_rows, args.sheet, refs=setting["refs"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
    keys = ("greedy_cer", "greedy_wer", "with_lm_cer", "with_lm_wer", "reference_greedy_cer", "hwd", "fid", "kid")
    print(json.dumps({k: row.get(k) for k in keys} | ({"control_hwd": row["control"].get("hwd")}
                                                      if "control" in row else {})), flush=True)


if __name__ == "__main__":
    main()
