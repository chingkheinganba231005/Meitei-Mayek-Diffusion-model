"""Render the generator's words, pseudo-writer by pseudo-writer, to shards.

    python scripts/render_data.py --word-repo /content/word_repo \
        --glyphs /content/drive/MyDrive/meitei-word-recognition/glyphs \
        --lexicon /content/drive/MyDrive/meitei-word-recognition/lexicon \
        --out /content/drive/MyDrive/meitei-word-diffusion/data

Each split uses its own characters and words (<glyphs>/<split>.npz, <lexicon>/<split>.tsv):
by default 75,000 training writers, 500 validation and 1,000 test writers, 16 words each.
Shards already written are kept, so an interrupted run carries on where it stopped.
--stats writes the counts of every split rendered so far (render.stats) to a .json file.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word-repo", required=True, help="the word-recognition repository at the pinned commit")
    ap.add_argument("--glyphs", required=True, help="folder with train.npz, val.npz and test.npz")
    ap.add_argument("--lexicon", required=True, help="folder with train.tsv, val.tsv and test.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--writers", nargs="+", default=["train=75000", "val=500", "test=1000"])
    ap.add_argument("--words", type=int, default=16, help="words per writer")
    ap.add_argument("--shard-writers", type=int, default=2000)
    ap.add_argument("--processes", type=int, default=os.cpu_count())
    ap.add_argument("--stats", help="write the counts of the rendered splits to this .json")
    args = ap.parse_args()
    sys.path.insert(0, args.word_repo)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mayek_diffusion.render import render_split, stats
    from mayek_words.synth import MEASURED

    writers = {k: int(v) for k, v in (w.split("=") for w in args.writers)}
    for split in args.splits:
        info = render_split(split, Path(args.glyphs) / f"{split}.npz", Path(args.lexicon) / f"{split}.tsv", MEASURED,
                            args.out, writers=writers[split], words=args.words,
                            shard_writers=min(args.shard_writers, writers[split]), processes=args.processes)
        print(json.dumps({k: info[k] for k in ("split", "writers", "words_per_writer", "minutes")}), flush=True)
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(stats(args.out), indent=1), encoding="utf-8")
        print(f"counts written to {args.stats}", flush=True)


if __name__ == "__main__":
    main()
