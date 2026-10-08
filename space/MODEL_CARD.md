---
license: {licence}
library_name: pytorch
pipeline_tag: text-to-image
tags:
  - handwriting-generation
  - meitei-mayek
  - manipuri
  - diffusion
---

# Handwritten Meitei Mayek word generation

The first diffusion model for handwritten Meitei Mayek, and the first learned generator of the
script's handwriting. It writes any word of everyday Meitei Mayek spelling in the hand of a
writer shown in one to eight reference word images, or in a random hand.

It denoises the word's image directly in pixels (EDM; a UNet of three levels on the 64-row
canvas folded 2 x 2 into channels), guided by the word printed in the font and stretched over
the width the handwriting should fill, by the word's characters, and by a style encoder over
the reference words. {parameters} M parameters; trained once, for {step} steps.

{results}

| File | What |
|---|---|
| `generator.safetensors` | the generator's weights (float16) |
| `config.json` | the model's configuration, the canvas, the alphabet and the sampling settings |
| `results.json` | the scores above, as the evaluation wrote them |

{owner}

## Use

The code is at [{repo_url}]({repo_url}); it needs the word synthesiser of the
[word recognition repository](https://github.com/chingkheinganba231005/meitei-mayek-word-recognition)
(commit `bdee9e7`) on the Python path for the font and the layout of the printed word.

```python
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from mayek_diffusion.canvas import from_real
from mayek_diffusion.release import load
from mayek_diffusion.sample import generate, paper
from mayek_diffusion.writers import Printer
from mayek_words.synth import load_priors

model, config = load(snapshot_download("{hf_id}"))
printer = Printer(load_priors())
refs = [from_real(np.asarray(Image.open(f).convert("L")))[0] for f in ["word1.png", "word2.png"]]
words = generate(model, ["ꯃꯤꯇꯩ", "ꯃꯌꯦꯛ"], refs, printer, steps=14, sampler="dpmpp", guidance=2.0)
Image.fromarray(paper(words[0])).save("word.png")
```

`refs=None` writes in a random hand. Each word comes out on a 64-row canvas with its baseline
at row 46 and letters 22 px high.

## Data

- **Characters:** the Tezpur University Meitei Mayek Handwritten Character Database (TUMMHCD;
  Hijam and Saharia, The Visual Computer 38:525-539, 2022), used under its terms. It has no
  writer information, so writers were made: each pseudo-writer keeps one word style and a small
  set of images per character across 16 words.
- **Words:** composed by the synthesiser of the word recognition project from a word list built
  from Meitei Wikipedia (CC BY-SA 4.0) and FineWeb-2 (ODC-By 1.0); only words are used, and
  they are not redistributed.
- **Real handwriting:** the author's own 100 words (CC BY 4.0), used only for adaptation and
  testing.

## Intended use and limits

Research on handwriting generation and recognition for Meitei Mayek, training data for
recognisers, and teaching material. The writers it learned from are synthetic, its letters are
22 px high (the source characters are 24 x 24), and its only real writer is the author, so
real hands are imitated less faithfully than synthetic ones. Words wider than 384 px on the
canvas were never seen in training.

**Do not imitate a person's handwriting without their consent**, and do not use generated
words to deceive.

## Licences

Code: MIT. Weights: {licence}. The font used for the printed word: Noto Sans Meetei Mayek, SIL
Open Font Licence 1.1. Real words: CC BY 4.0. TUMMHCD: character images from Tezpur
University's database, used under its terms.

## Citation

```bibtex
@misc{{rajkumar2026generation,
  author       = {{Rajkumar, Chingkheinganba}},
  title        = {{Handwritten {{Meitei Mayek}} word generation}},
  year         = {{2026}},
  howpublished = {{\url{{{repo_url}}}}}
}}
```
