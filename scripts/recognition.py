"""Generated words for the word recogniser (E6).

    # 20,000 training-lexicon words in the owner's hand (few-shot, the 4 fixed references)
    python scripts/recognition.py generate --word-repo /content/word_repo --weights .../runs/main/final.pt \\
        --setting .../results/eval_val_grid.json --owner .../data/owner \\
        --lexicon /content/lexicon/train.tsv --out /content/e6/generated.tar

    # the deployed recogniser trained further: (i) synthesiser words only, (ii) half generated
    python scripts/recognition.py finetune --word-repo /content/word_repo --recogniser /content/recogniser \\
        --glyphs /content/glyphs/train.npz --lexicon /content/lexicon/train.tsv --out .../runs/e6_synth.pt
    python scripts/recognition.py finetune ... --generated /content/e6/generated.tar --out .../runs/e6_mixed.pt

    # the three on the 96 real words that are not references, and on the synthetic test words
    python scripts/recognition.py score --word-repo /content/word_repo --recogniser /content/recogniser \\
        --models start=/content/recogniser/recogniser.pt synth=.../e6_synth.pt mixed=.../e6_mixed.pt \\
        --real /content/word_repo/real_words --split .../results/owner_split.json \\
        --synth-test /content/synth/test.tar --work /content/e6 --out .../results/e6_recognition.json

generate: texts drawn as the synthesiser draws them (mayek_words.synth.Words.text, seed 300,
10% scrambled), written by the generator with the setting chosen on validation (or --sampler
and --steps), cut to their ink as the real words were.

finetune: 5,000 steps of 64 words, lr 1e-4 after 200 warm-up steps, the recogniser's own
augmentation, CTC and weight average; the synthetic words are drawn by the recogniser's
latest recipe (ꯥ ꯩ ꯪ beside their letter in half of the words, 10% scrambled words) from
--data-seed, which no earlier training used.

score: greedy and language-model CER and WER of each model (the language model's weights from
decoding.json), with 95% bootstrap intervals; the real set is small. The real words' sign
positions informed the recogniser's latest synthesiser setting, which the results note.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["generate", "finetune", "score"])
    ap.add_argument("--word-repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", help="generate: the generator (final.pt or a release folder)")
    ap.add_argument("--setting", help="generate: the validation grid's results .json")
    ap.add_argument("--sampler", choices=["heun", "dpmpp"])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--owner", help="generate: the prepared owner folder (its 4 fixed references)")
    ap.add_argument("--lexicon", help="the training lexicon (.tsv)")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--recogniser", help="folder with recogniser.pt, char_lm.pkl, decoding.json")
    ap.add_argument("--glyphs", help="finetune: the training characters (.npz)")
    ap.add_argument("--generated", help="finetune: generated words (.tar) for half of each batch")
    ap.add_argument("--train-steps", type=int, default=5000)
    ap.add_argument("--data-seed", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) - 2, 1))
    ap.add_argument("--models", nargs="+", help="score: name=checkpoint ...")
    ap.add_argument("--real", help="score: the real word set")
    ap.add_argument("--split", help="score: owner_split.json (its references are left out)")
    ap.add_argument("--synth-test", help="score: the synthetic test set (.tar)")
    ap.add_argument("--work", help="score: a local folder")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sys.path.insert(0, args.word_repo)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.command == "generate":
        import torch

        from mayek_diffusion.owner import references
        from mayek_diffusion.protocol import chosen_setting
        from mayek_diffusion.recognition import draw_texts, write_generated
        from mayek_diffusion.release import load_weights
        from mayek_diffusion.writers import Printer
        from mayek_words.lexicon import Lexicon
        from mayek_words.synth import load_priors

        s = chosen_setting(args.setting if args.setting and Path(args.setting).exists() else None)
        s.update({k: getattr(args, k) for k in ("sampler", "steps") if getattr(args, k)})
        printer = Printer(load_priors())
        texts = draw_texts(Lexicon.load(args.lexicon), args.n, args.seed, printer=printer)
        model, _ = load_weights(args.weights, s["sigma_rel"], args.device)
        dtype = torch.bfloat16 if args.device.startswith("cuda") else None
        write_generated(model, texts, references(args.owner), printer, args.out,
                        batch=args.batch, steps=s["steps"], sampler=s["sampler"], guidance=s["guidance"],
                        interval=tuple(s["interval"]), device=args.device, dtype=dtype)
        Path(str(args.out) + ".json").write_text(json.dumps({"words": len(texts), "seed": args.seed, "setting": s,
                                                             "weights": str(args.weights)}, indent=1))
        print(json.dumps({"words": len(texts), "out": str(args.out)}))
        return

    if args.command == "finetune":
        from mayek_diffusion.recognition import finetune_recogniser
        from mayek_words.glyphs import GlyphStore
        from mayek_words.lexicon import Lexicon
        from mayek_words.synth import Config, Words, WordSynth, load_priors

        synth = WordSynth(GlyphStore.load(args.glyphs), load_priors(), Config(p_marks_beside=0.5))
        words = Words(synth, Lexicon.load(args.lexicon), seed=args.data_seed, scrambled=0.1)
        h = finetune_recogniser(Path(args.recogniser) / "recogniser.pt", words, args.out, generated=args.generated,
                                steps=args.train_steps, workers=args.workers, device=args.device)
        print(json.dumps({"out": str(args.out), "last": h[-1] if h else None}))
        return

    # score
    from mayek_diffusion.recognition import real_set
    from mayek_diffusion.scores import bootstrap, read_predictions, recognise, recogniser_files

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    n_real = real_set(args.real, split["files"]["references"], work / "real.tar")
    sets = {"real": work / "real.tar"}
    if args.synth_test:
        sets["synthetic_test"] = Path(args.synth_test)
    base = recogniser_files(args.recogniser)
    res = {"real_words": n_real, "left_out": split["files"]["references"], "models": {},
           "note": "The real words' sign positions informed the recogniser's latest synthesiser setting "
                   "(signs above also written beside their letter)."}
    for spec in args.models:
        name, ck = spec.split("=", 1)
        res["models"][name] = {"checkpoint": ck}
        for set_name, path in sets.items():
            pred = work / f"{name}_{set_name}.tsv"
            r = recognise(path, dict(base, checkpoint=ck), args.word_repo, work / f"{name}_{set_name}.json",
                          predictions=pred, device=args.device)
            row = {"greedy": r["greedy"], "with_lm": r["with_lm"]}
            for col in ("greedy", "with_lm"):
                refs, hyps = read_predictions(pred, col)
                row[f"{col}_bootstrap"] = bootstrap(refs, hyps)
            res["models"][name][set_name] = row
            print(f"{name} on {set_name}: CER {r['with_lm']['cer']:.4f}, WER {r['with_lm']['wer']:.4f} "
                  f"(greedy {r['greedy']['cer']:.4f}, {r['greedy']['wer']:.4f})", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
