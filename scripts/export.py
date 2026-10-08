"""The release: the chosen generator, its cards and the demo, ready to upload.

    python scripts/export.py --word-repo /content/word_repo \\
        --weights /content/drive/MyDrive/meitei-word-diffusion/runs/main/final.pt \\
        --results /content/drive/MyDrive/meitei-word-diffusion/results \\
        --data /content/drive/MyDrive/meitei-word-diffusion/data \\
        --owner-weights /content/drive/MyDrive/meitei-word-diffusion/runs/owner/owner.pt \\
        --out /content/drive/MyDrive/meitei-word-diffusion/release

Writes <out>/model (the model repository: generator.safetensors in float16, config.json,
results.json, README.md as the model card, presets/) and <out>/space (the Gradio Space: app.py,
requirements.txt, README.md as the Space card, presets/, and the code it runs: mayek_diffusion
and the word repository's mayek_words with its font).

The average of the weights and the sampling settings are those chosen on validation
(<results>/eval_val_grid.json); without it, the defaults. "The author's hand" uses the adapted
generator (exported as owner_generator.safetensors) if results/owner.json shows it better by
the validation rule (the lowest HWD among those whose greedy CER is at most the real words'
plus 1 point), and otherwise the main generator with the four fixed references. The presets
are six test writers (half of them writing ꯥ ꯩ ꯪ beside their letter), four references each,
and the four fixed references of the author.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word-repo", required=True)
    ap.add_argument("--weights", required=True, help="the main run's final.pt")
    ap.add_argument("--results", required=True, help="folder of the results files")
    ap.add_argument("--data", required=True, help="the rendered splits (test, and owner/ if prepared)")
    ap.add_argument("--owner-weights", help="the adapted generator (owner.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-id", default="Chingkheinganba/handwritten-meitei-mayek-word-generation")
    ap.add_argument("--licence", default="mit", help="of the weights (Hugging Face licence id)")
    ap.add_argument("--presets", type=int, default=6)
    args = ap.parse_args()
    sys.path.insert(0, args.word_repo)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np

    from mayek_diffusion.cards import cards, load_results, results_json
    from mayek_diffusion.data import WordSet
    from mayek_diffusion.demo import save_canvases
    from mayek_diffusion.owner import references
    from mayek_diffusion.protocol import chosen_setting
    from mayek_diffusion.release import SAMPLING, export, load, load_weights, pick_presets, vendor
    from mayek_diffusion.scores import choose

    results_dir, out = Path(args.results), Path(args.out)
    model_dir, space_dir = out / "model", out / "space"
    shutil.rmtree(model_dir, ignore_errors=True)
    model_dir.mkdir(parents=True)
    grid = results_dir / "eval_val_grid.json"
    setting = chosen_setting(grid if grid.exists() else None)
    model, info = load_weights(args.weights, setting["sigma_rel"])
    owner_use = None
    owner_res = results_dir / "owner.json"
    if args.owner_weights and Path(args.owner_weights).exists() and owner_res.exists():
        o = json.loads(owner_res.read_text(encoding="utf-8"))
        rows = [{"name": k, "greedy_cer": o[k]["recognised"]["greedy"]["cer"], "hwd": o[k].get("hwd")}
                for k in ("few_shot", "adapted")]
        owner_use = choose(rows, o["real"]["greedy"]["cer"])["name"]
    sampling = {**{k: setting[k] for k in ("sigma_rel", "guidance", "interval", "refs", "sampler", "steps")},
                "demo_sampler": SAMPLING["demo_sampler"], "demo_steps": SAMPLING["demo_steps"],
                "chosen_on": "eval_val_grid.json" if grid.exists() else "defaults", "owner": owner_use}
    source = {"weights": Path(args.weights).name, "run": Path(args.weights).parent.name, "step": info.get("step"),
              "sigma_rel": setting["sigma_rel"]}
    export(model, model_dir, sampling=sampling, prior=info.get("prior") is not False, source=source)
    if owner_use == "adapted":
        export(load_weights(args.owner_weights)[0], model_dir, name="owner_generator")
    _, config = load(model_dir)

    presets = model_dir / "presets"
    test = WordSet(Path(args.data) / "test")
    per = test.per_writer
    for k, w in enumerate(pick_presets(test, args.presets), 1):
        save_canvases([test.target(i).astype(np.float32) / 255 for i in range(w * per, w * per + 4)],
                      presets / f"writer_{k}")
    owner_dir = Path(args.data) / "owner"
    if (owner_dir / "adapt" / "info.json").exists():
        save_canvases(references(owner_dir), presets / "author")

    results = load_results(results_dir)
    model_card, space_card = cards(results, config, args.hf_id, args.licence)
    (model_dir / "README.md").write_text(model_card, encoding="utf-8")
    (model_dir / "results.json").write_text(json.dumps(results_json(results, config), indent=1, ensure_ascii=False),
                                            encoding="utf-8")

    shutil.rmtree(space_dir, ignore_errors=True)
    space_dir.mkdir(parents=True)
    for f in ("app.py", "requirements.txt"):
        shutil.copyfile(root / "space" / f, space_dir / f)
    (space_dir / "README.md").write_text(space_card, encoding="utf-8")
    shutil.copytree(presets, space_dir / "presets")
    vendor(space_dir, args.word_repo)
    for d in (model_dir, space_dir):
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        print(f"{d}: {size / 1e6:.1f} MB", flush=True)
    print(json.dumps({"setting": sampling, "presets": sorted(p.name for p in presets.iterdir())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
