"""The owner's hand (E5): few-shot imitation from four real words, and adaptation to 70.

    python scripts/owner.py prepare --word-repo /content/word_repo --real /content/word_repo/real_words \\
        --out /content/drive/MyDrive/meitei-word-diffusion/data/owner \\
        --split /content/drive/MyDrive/meitei-word-diffusion/results/owner_split.json

    python scripts/owner.py adapt --word-repo /content/word_repo --weights .../runs/main/final.pt \\
        --setting .../results/eval_val_grid.json --last .../runs/main/last.pt \\
        --owner .../data/owner --data .../data --out .../runs/owner/owner.pt

    python scripts/owner.py score --word-repo /content/word_repo --weights .../runs/main/final.pt \\
        --adapted .../runs/owner/owner.pt --setting .../results/eval_val_grid.json --owner .../data/owner \\
        --data .../data --recogniser /content/recogniser --work /content/eval/owner \\
        --out .../results/owner.json --sheet .../eval/owner.png

prepare: the real words split once with numpy.random.default_rng(2026).permutation(100): the
first 70 for adaptation (the first 4 of them are the fixed few-shot references), the last 30
held out; written as one-writer shard sets (owner/adapt, owner/held_out) and the split as a
.json file.

adapt: the chosen average of the main run trained further for 1,500 steps at lr 2e-5 (100
warm-up steps), batches of the adaptation words (their references: other adaptation words)
alternating with batches of synthetic training words (--train-shards shards of the training
split), no contrastive term; the result is a plain average of the weights (decay 0.999).
About 15 minutes on an A100.

score: (a) the main model with the 4 references and (b) the adapted model with the same
references write the 30 held-out texts (--seeds samples each); for both: CER and WER through
the recogniser (greedy and with the language model), HWD and KID against the 30 real
held-out words, and the recogniser on those real words (the reference level). Then both
models on --val-writers validation writers with the chosen setting, to check that the
adapted model still writes other hands.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def setting_of(path):
    from mayek_diffusion.protocol import chosen_setting
    return chosen_setting(path if path and Path(path).exists() else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "adapt", "score"])
    ap.add_argument("--word-repo", required=True)
    ap.add_argument("--real", help="prepare: the real word set (images/, labels.tsv)")
    ap.add_argument("--split", help="prepare: the split, .json")
    ap.add_argument("--owner", help="the prepared folder (adapt/, held_out/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", help="the main run's final.pt (or a release folder)")
    ap.add_argument("--setting", help="the validation grid's results .json (the chosen setting)")
    ap.add_argument("--last", help="adapt: the main run's last.pt (its loss weighting and micro-batches)")
    ap.add_argument("--data", help="the rendered splits (train for adapt; val for score)")
    ap.add_argument("--train-shards", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--adapted", help="score: the adapted generator (owner.pt)")
    ap.add_argument("--recogniser", help="score: folder with recogniser.pt, char_lm.pkl, decoding.json")
    ap.add_argument("--work", help="score: a local folder for the images")
    ap.add_argument("--sheet", help="score: real | few-shot | adapted, .png")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--val-writers", type=int, default=100)
    ap.add_argument("--no-style", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true", help="adapt: small batches (dry runs on CPU)")
    args = ap.parse_args()
    sys.path.insert(0, args.word_repo)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch

    from mayek_diffusion.writers import Printer
    from mayek_words.synth import load_priors

    printer = Printer(load_priors())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        from mayek_diffusion.owner import prepare
        s = prepare(args.real, out, printer)
        Path(args.split).parent.mkdir(parents=True, exist_ok=True)
        Path(args.split).write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"adapt": len(s["adapt"]), "references": s["files"]["references"],
                          "held_out": len(s["held_out"]), "ink_lost_max": s["ink_lost_max"]}, ensure_ascii=False))
        return

    from mayek_diffusion.data import WordSet
    from mayek_diffusion.owner import references
    from mayek_diffusion.release import load_weights

    setting = setting_of(args.setting)
    if args.command == "adapt":
        from mayek_diffusion.finetune import AdaptConfig, adapt, checkpoint
        model, info = load_weights(args.weights, setting["sigma_rel"])
        logvar, accum = None, 1
        if args.last and Path(args.last).exists():
            last = torch.load(args.last, map_location="cpu", weights_only=False)
            logvar, accum = last["logvar"], max(int(last.get("accum") or 1), 1)
            del last
        cfg = AdaptConfig(steps=args.steps)
        if args.tiny:
            cfg.owner_pixels, cfg.synth_pixels, cfg.warmup, cfg.log_every = 8 * 128, 8 * 128, 2, 2
        train = WordSet(Path(args.data) / "train", limit_shards=args.train_shards) if args.data else None
        avg, history = adapt(model, WordSet(Path(args.owner) / "adapt"), train, cfg, logvar, args.device, accum)
        source = {"from": str(args.weights), "sigma_rel": setting["sigma_rel"], "step": info.get("step")}
        torch.save(checkpoint(avg.cpu(), cfg, history, source), out)
        print(json.dumps({"steps": cfg.steps, "last": history[-1] if history else None, "out": str(out)}))
        return

    # score
    from mayek_diffusion.evaluate import write_fixed_set, write_writer_folders
    from mayek_diffusion.protocol import Evaluator
    from mayek_diffusion.sample import generate
    from mayek_diffusion.scores import StyleScorer, recognise
    from mayek_diffusion.sheets import grid

    rec, work = Path(args.recogniser), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    read = lambda tar, o: recognise(tar, rec, args.word_repo, o, device=args.device)
    scorer = None if args.no_style else StyleScorer()
    held = WordSet(Path(args.owner) / "held_out")
    refs = references(args.owner)
    texts = [str(t) for t in held.texts]
    real = [held.target(i).astype(np.float32) / 255 for i in range(len(held))]
    dtype = torch.bfloat16 if args.device.startswith("cuda") else None
    write_fixed_set(real, texts, work / "real.tar")
    shutil.rmtree(work / "real", ignore_errors=True)
    write_writer_folders(real, [0] * len(real), work / "real")
    res = {"setting": setting, "seeds": args.seeds, "held_out": len(texts), "references": len(refs),
           "real": read(work / "real.tar", work / "real.json")}
    gens = {}
    for name, path in (("few_shot", args.weights), ("adapted", args.adapted)):
        model, _ = load_weights(path, setting["sigma_rel"], args.device)
        gen = []
        for seed in range(args.seeds):
            gen += generate(model, texts, refs, printer, steps=setting["steps"], sampler=setting["sampler"],
                            guidance=setting["guidance"], interval=tuple(setting["interval"]), seed=seed,
                            device=args.device, dtype=dtype)
        gens[name] = gen
        write_fixed_set(gen, texts * args.seeds, work / f"{name}.tar")
        row = {"recognised": read(work / f"{name}.tar", work / f"{name}.json")}
        if scorer is not None:
            shutil.rmtree(work / name, ignore_errors=True)
            write_writer_folders(gen, [0] * len(gen), work / name)
            row.update(scorer(work / name, work / "real", ("hwd", "kid")))
        res[name] = row
        del model
    if args.val_writers and args.data:
        val = WordSet(Path(args.data) / "val")
        style = None if scorer is None else (lambda f, r, w: scorer(f, r, w))
        for name, path in (("few_shot", args.weights), ("adapted", args.adapted)):
            ev = Evaluator(path, val, printer, range(args.val_writers), read, style, work / f"val_{name}",
                           device=args.device)
            r = ev.run(setting, max(4, setting["refs"]), val.per_writer - max(4, setting["refs"]))
            res[name]["validation_writers"] = {k: r.get(k) for k in ("writers", "words", "greedy_cer", "greedy_wer",
                                                                     "with_lm_cer", "hwd", "reference_greedy_cer")}
    if args.sheet:
        n = min(12, len(texts))
        grid([[[real[i]], [gens["few_shot"][i + s * len(texts)] for s in range(args.seeds)],
               [gens["adapted"][i + s * len(texts)] for s in range(args.seeds)]] for i in range(n)], args.sheet)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    brief = {k: {"greedy_cer": res[k]["recognised"]["greedy"]["cer"], "hwd": res[k].get("hwd")}
             for k in ("few_shot", "adapted")}
    print(json.dumps({"real_greedy_cer": res["real"]["greedy"]["cer"], **brief}))


if __name__ == "__main__":
    main()
