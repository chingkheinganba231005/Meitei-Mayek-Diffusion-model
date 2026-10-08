"""End-to-end check of training and sampling, for the smoke test before the main run.

    python scripts/overfit_check.py --word-repo /content/word_repo \
        --data /content/drive/MyDrive/meitei-word-diffusion/data --out overfit.png

A small generator (--full: the real configuration) memorises one batch of 8 words (fresh
noise levels and noise every step, no condition dropping), then writes those words from pure
noise with both samplers. The ink of the result must overlap the real words' ink (IoU) by at
least --min-iou, and by more than twice as much as a control in which every word gets another
word's text and prior. On CPU the small model, 600 steps, gave IoU 0.82 (Heun) and 0.79
(DPM-Solver++) against 0.16 for the control. A failure means something in training or
sampling is broken: do not start the main run. On an A100 it takes about a minute.
--min-iou 0 only runs the steps (tests).
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word-repo", required=True)
    ap.add_argument("--data", required=True, help="folder with train/ (scripts/render_data.py)")
    ap.add_argument("--out", required=True, help="sheet .png: real | Heun | DPM-Solver++")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--min-iou", type=float, default=0.7)
    ap.add_argument("--full", action="store_true", help="the real model configuration instead of the small one")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sys.path.insert(0, args.word_repo)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from PIL import Image

    from mayek_diffusion.data import Batcher, WordSet
    from mayek_diffusion.edm import EDM, LogVar, dpmpp_2m, heun, karras_sigmas
    from mayek_diffusion.model import Generator, ModelConfig
    from mayek_diffusion.train import TrainConfig, step_losses, to_device

    torch.manual_seed(0)
    dev = torch.device(args.device)
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    ds = WordSet(Path(args.data) / "train", limit_shards=1)
    batcher = Batcher(ds, pixels=8 * 128, refs=4, seed=3, p_all_refs=1.0)
    b = to_device(batcher.sample(128 if 128 in batcher.buckets else int(batcher.Ws[0])), dev)
    cfg = TrainConfig(p_drop_style=0.0, p_drop_both=0.0)
    mc = ModelConfig() if args.full else ModelConfig(channels=(64, 128, 128), d=128, text_layers=2, style_layers=1)
    model, logvar, edm = Generator(mc).to(dev), LogVar().to(dev), EDM()
    params = list(model.parameters()) + list(logvar.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4 if args.full else 1e-3, betas=(0.9, 0.99))
    for s in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = opt.defaults["lr"] * min(1.0, (s + 1) / 50)
        with amp:
            loss, terms = step_losses(model, logvar, edm, b, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (s + 1) % 100 == 0:
            print(f"step {s + 1}: weighted mse {float(terms['wmse']):.4f}", flush=True)
    model.eval()
    y = b["target"]
    ink_y = (y + 1) / 2 > 0.5
    iou = lambda x: float((((x + 1) / 2 > 0.5) & ink_y).sum() / (((x + 1) / 2 > 0.5) | ink_y).sum())
    g = torch.Generator().manual_seed(1)
    noise = lambda: torch.randn(y.shape, generator=g).to(dev)

    def sample(fn, steps, text, prior):
        with torch.no_grad(), amp:
            cond = model.condition(text, b["ref"], b["ref_width"], b["ref_mask"])
        net = lambda x_in, c_noise: model(x_in, prior, c_noise, cond)

        def den(x, sig):
            with torch.no_grad(), amp:
                return edm.denoise(net, x, torch.full((x.shape[0],), sig, device=dev))
        sig = karras_sigmas(steps)
        return fn(den, noise() * sig[0], sig).float()

    x_heun = sample(heun, 18, b["text"], b["prior"])
    x_dpm = sample(dpmpp_2m, 16, b["text"], b["prior"])
    perm = torch.roll(torch.arange(y.shape[0], device=dev), 1)
    x_ctl = sample(heun, 18, b["text"][perm], b["prior"][perm])
    result = {"iou_heun18": round(iou(x_heun), 3), "iou_dpmpp16": round(iou(x_dpm), 3),
              "iou_control": round(iou(x_ctl), 3), "min_iou": args.min_iou}
    result["passed"] = bool(args.min_iou <= 0 or (min(result["iou_heun18"], result["iou_dpmpp16"]) >= args.min_iou
                                                  and result["iou_control"] < 0.5 * result["iou_heun18"]))
    rows = [((t + 1) / 2).clamp(0, 1)[:, 0].cpu().numpy() for t in (y, x_heun, x_dpm)]
    bar = np.full((rows[0].shape[1], 4), 0.5, np.float32)
    sheet = np.concatenate([np.concatenate([rows[0][i], bar, rows[1][i], bar, rows[2][i]], 1)
                            for i in range(len(rows[0]))], 0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(255 - (sheet * 255).astype(np.uint8)).save(args.out)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit("the generator did not give the memorised words back: do not start the main run")


if __name__ == "__main__":
    main()
