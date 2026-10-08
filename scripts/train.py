"""Train the generator, or resume its run, against a time budget.

    python scripts/train.py --word-repo /content/word_repo \
        --data /content/drive/MyDrive/meitei-word-diffusion/data \
        --run /content/drive/MyDrive/meitei-word-diffusion/runs/main --hours 32

The number of steps is set from the speed of the first steps (mayek_diffusion.train). Running
the same command again resumes from <run>/last.pt. A sample sheet of validation writers is
drawn at every checkpoint (<run>/samples/) and at the end (<run>/samples/final.png).
--tiny trains a very small model on small batches, for dry runs on CPU.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before torch starts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word-repo", required=True)
    ap.add_argument("--data", required=True, help="folder with train/ and val/ (scripts/render_data.py)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--local", default="/content/ckpt", help="local folder for checkpoints before copying")
    ap.add_argument("--hours", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=0, help="total steps (default: from the measured speed)")
    ap.add_argument("--stop-at", type=int, default=0, help="stop at this step (the ablation)")
    ap.add_argument("--channels", type=int, nargs=3, default=[128, 256, 512])
    ap.add_argument("--no-prior", action="store_true", help="leave the printed prior out (the ablation)")
    ap.add_argument("--save-minutes", type=float, default=20.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, args.word_repo)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch

    from mayek_diffusion.data import WordSet
    from mayek_diffusion.model import Generator, ModelConfig
    from mayek_diffusion.sample import sheet
    from mayek_diffusion.train import TrainConfig, train
    from mayek_diffusion.writers import Printer
    from mayek_words.synth import load_priors

    cuda = args.device.startswith("cuda")
    cfg = TrainConfig(hours=args.hours, steps=args.steps, stop_at=args.stop_at, prior=not args.no_prior,
                      save_minutes=args.save_minutes)
    mcfg = ModelConfig(channels=tuple(args.channels))
    if args.tiny:
        cfg.pixels, cfg.warmup, cfg.log_every, cfg.val_every = 8 * 128, 2, 2, 4
        mcfg = ModelConfig(channels=(32, 64, 64), d=64, text_layers=1, style_layers=1)
    printer, val = Printer(load_priors()), WordSet(Path(args.data) / "val")
    dtype = torch.bfloat16 if cuda else None
    draw = lambda model, step, path: sheet(model, val, printer, path, device=args.device, dtype=dtype)
    result = train(cfg, args.data, args.run, mcfg, device=args.device, local_dir=args.local, sheet=draw)
    if result["done"]:
        final = torch.load(Path(args.run) / "final.pt", map_location="cpu", weights_only=False)
        mc = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in final["model_cfg"].items()})
        model = Generator(mc)
        model.load_state_dict(final["ema"][0.10])
        draw(model.to(args.device), result["step"], Path(args.run) / "samples" / "final.png")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
