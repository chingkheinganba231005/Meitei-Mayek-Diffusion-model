"""Handwritten Meitei Mayek words, written by the generator in a chosen hand (Gradio, CPU).

The generator and its config.json come from the model repository (MODEL_ID), or from a local
folder (MODEL_DIR). mayek_diffusion and mayek_words (with its font) sit next to this file.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from mayek_diffusion.demo import (compose_line, enlarge, keyboard, load_presets, parse,  # noqa: E402
                                  references_from_images)
from mayek_diffusion.release import load  # noqa: E402
from mayek_diffusion.sample import generate  # noqa: E402
from mayek_diffusion.writers import Printer  # noqa: E402
from mayek_words.synth import load_priors  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "Chingkheinganba/handwritten-meitei-mayek-word-generation")
AUTHOR = "The author's hand"
RANDOM = "A random hand"
PRESET = "A writer of the test set"
UPLOAD = "Your own word images"


def model_folder():
    if os.environ.get("MODEL_DIR"):
        return Path(os.environ["MODEL_DIR"])
    from huggingface_hub import hf_hub_download

    files = ["config.json", "generator.safetensors"]
    folder = None
    for f in files:
        folder = Path(hf_hub_download(MODEL_ID, f)).parent
    try:
        hf_hub_download(MODEL_ID, "owner_generator.safetensors")
    except Exception:            # the adapted generator is optional
        pass
    return folder


torch.set_num_threads(int(os.environ.get("THREADS", "2")))
FOLDER = model_folder()
MODEL, CONFIG = load(FOLDER)
SAMPLING = CONFIG["sampling"]
OWNER = None
if SAMPLING.get("owner") == "adapted" and (FOLDER / "owner_generator.safetensors").exists():
    OWNER = load(FOLDER, "owner_generator")[0]
PRINTER = Printer(load_priors())
PRESETS = load_presets(HERE / "presets")
AUTHOR_REFS = PRESETS.pop("author", None)
CHOICES = ([AUTHOR] if AUTHOR_REFS else []) + [PRESET, UPLOAD, RANDOM]


def style(source, preset, uploads):
    """-> (model, references or None, messages)."""
    if source == AUTHOR:
        return (OWNER or MODEL), AUTHOR_REFS, []
    if source == PRESET:
        return MODEL, PRESETS[preset], []
    if source == UPLOAD:
        images = [np.asarray(Image.open(getattr(f, "name", f))) for f in (uploads or [])][:8]
        refs, notes = references_from_images(images)
        if not refs:
            raise gr.Error("Upload 1 to 8 images, each of one handwritten word (dark ink on light paper).")
        return MODEL, refs, notes
    return MODEL, None, []


def show_refs(source, preset, uploads):
    try:
        _, refs, notes = style(source, preset, uploads)
    except gr.Error:
        return [], ""
    return [enlarge(r, 2) for r in (refs or [])], "\n".join(notes)


def write(text, source, preset, uploads, guidance, steps, width, seed, samples):
    try:
        words = parse(text)
    except ValueError as e:
        raise gr.Error(str(e))
    model, refs, notes = style(source, preset, uploads)
    t0, lines = time.time(), []
    for k in range(int(samples)):
        canvases = generate(model, words, refs, PRINTER, steps=int(steps), sampler=SAMPLING["demo_sampler"],
                            guidance=float(guidance), interval=tuple(SAMPLING["interval"]), width_scale=float(width),
                            seed=int(seed) + k, device="cpu")
        lines.append(compose_line(canvases))
    W = max(x.shape[1] for x in lines)                  # the samples one above the other
    image = enlarge(np.concatenate([np.pad(x, ((0, 6), (0, W - x.shape[1]))) for x in lines]), 3)
    path = Path(tempfile.mkdtemp()) / "meitei_mayek.png"
    Image.fromarray(image).save(path)
    took = f"{time.time() - t0:.0f} s for {len(words) * int(samples)} word(s)."
    return image, str(path), "\n".join(notes + [took])


def build():
    with gr.Blocks(title="Handwritten Meitei Mayek word generation") as demo:
        gr.Markdown(
            "# Handwritten Meitei Mayek word generation\n"
            "Type up to six Meitei Mayek words and choose a hand: one of the test writers, the author's own, "
            "a random one, or yours from one to eight photos or scans of words you have written. The generator "
            "is the first diffusion model for Meitei Mayek handwriting. It runs here on two CPU cores, so a word "
            "takes about 10 to 20 seconds. Please do not imitate a person's handwriting without their consent.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", placeholder="ꯃꯤꯇꯩ ꯃꯌꯦꯛ", lines=1)
                with gr.Accordion("Keyboard", open=True):
                    for name, chars in keyboard():
                        gr.Markdown(f"**{name}**")
                        with gr.Row():
                            for ch in chars:
                                b = gr.Button(ch, size="sm", min_width=36)
                                b.click(lambda t, c=ch: (t or "") + c, inputs=text, outputs=text)
                    with gr.Row():
                        gr.Button("Space", size="sm").click(lambda t: (t or "") + " ", inputs=text, outputs=text)
                        gr.Button("Delete", size="sm").click(lambda t: (t or "")[:-1], inputs=text, outputs=text)
                        gr.Button("Clear", size="sm").click(lambda: "", outputs=text)
            with gr.Column():
                source = gr.Radio(CHOICES, value=CHOICES[0], label="Hand")
                preset = gr.Dropdown(list(PRESETS), value=next(iter(PRESETS), None), label="Test writer")
                uploads = gr.File(file_count="multiple", file_types=["image"], label="Word images (1 to 8)")
                refs = gr.Gallery(label="References as the generator sees them", columns=4, height=180,
                                  object_fit="contain")
                notes_in = gr.Markdown()
        with gr.Accordion("Settings", open=False):
            guidance = gr.Slider(1.0, 4.0, value=float(SAMPLING["guidance"]), step=0.25, label="Guidance")
            steps = gr.Slider(8, 30, value=int(SAMPLING["demo_steps"]), step=1, label="Steps")
            width = gr.Slider(0.8, 1.25, value=1.0, step=0.05, label="Width")
            seed = gr.Number(value=0, precision=0, label="Seed")
            samples = gr.Slider(1, 4, value=1, step=1, label="Samples")
        go = gr.Button("Write", variant="primary")
        out = gr.Image(label="Written", type="numpy", interactive=False, show_label=True)
        download = gr.File(label="PNG")
        notes = gr.Markdown()
        for comp in (source, preset, uploads):
            comp.change(show_refs, inputs=[source, preset, uploads], outputs=[refs, notes_in])
        demo.load(show_refs, inputs=[source, preset, uploads], outputs=[refs, notes_in])
        go.click(write, inputs=[text, source, preset, uploads, guidance, steps, width, seed, samples],
                 outputs=[out, download, notes])
    return demo


if __name__ == "__main__":
    build().launch()
