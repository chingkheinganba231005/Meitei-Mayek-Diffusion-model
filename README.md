# Handwritten Meitei Mayek word generation

The first diffusion model for handwritten Meitei Mayek, and the first learned generator of the
script's handwriting. It writes any word of everyday Meitei Mayek spelling in the hand of a
writer shown in one to eight reference word images. It is trained without a single real
handwritten word: its writers are pseudo-writers composed from the isolated characters of the
Tezpur University Meitei Mayek Handwritten Character Database (TUMMHCD), which has no writer
information.

**Status.** The code, the notebooks and the tests are complete and run on CPU; the generator
has not been trained yet. The trained model, a demo and the results will follow, with every
number written to `results/` by the code.

## How it works

1. **The canvas.** Every word image, generated or real, lies on one canvas: 64 rows, ink 1 on
   paper 0, the baseline at row 46 and letters 22 px high, any width (at most 384 px). Real
   words are placed by the letters' band measured on their ink (`mayek_diffusion/canvas.py`).
2. **Pseudo-writers** (`writers.py`, `render.py`). The word synthesiser of the
   [word recognition project](https://github.com/chingkheinganba231005/meitei-mayek-word-recognition)
   composes TUMMHCD characters into words. Here a writer keeps one word style (letter size and
   width, spacing, slant, pen, how often letters join, and whether ꯥ ꯩ ꯪ are written beside
   their letter) and, for every character, two images among those closest to its style, across
   its 16 words. 75,000 training writers give 1.2 million words, rendered once to shards.
3. **Content.** The word printed in the font and stretched over the width the handwriting
   should fill, as a second input channel: it says where every letter and sign goes, which is
   the hard part of a script that stacks signs above, below and beside its letters. And the
   word's characters, as tokens for cross-attention.
4. **Style.** A small CNN and transformer over the reference words (any number, as a set)
   give style tokens for cross-attention, a global style vector, the writer's width relative
   to print, and a projection trained with a supervised contrastive loss.
5. **Denoiser** (`model.py`). The diffusers `UNet2DConditionModel` on the canvas folded 2 x 2
   into channels: three levels of 128, 256 and 512 channels, attention at the two coarser
   ones. 113.5 M parameters in all.
6. **Diffusion** (`edm.py`). EDM (Karras et al. 2022) with the learned loss weighting of EDM2;
   classifier-free guidance on the style, applied in an interval of noise levels; Heun's method
   or DPM-Solver++(2M) for sampling.
7. **Training** (`train.py`). One run against a budget of 32 GPU-hours, its length in steps set
   from the speed of its first steps, resumable across Colab sessions, keeping three
   power-function averages of the weights (EDM2) to choose from afterwards.
8. **Evaluation** (`protocol.py`, `scores.py`). The deployed word recogniser reads the
   generated words (CER, WER); HWD compares them with the real words of the same writers.
   The average, the guidance, the sampler and the number of references are chosen on
   validation writers by a rule fixed in advance; the test writers are scored once.
9. **The author's hand** (`owner.py`, `finetune.py`). The 100 real words of the word
   recognition project: few-shot from four of them, and adaptation on seventy, scored on the
   thirty held out.

## Running it

The notebooks run on Google Colab with a working folder on Google Drive (at least 12 GB free).
Run them in order; each says what it needs, can be run again to resume, and ends by
disconnecting the runtime.

| Notebook | Runtime | GPU hours | What it does |
|---|---|---:|---|
| `notebooks/1_render_data.ipynb` | CPU | 0 | renders the training, validation and test writers |
| `notebooks/2_train.ipynb` | A100 | 32.7 | the checks (overfit, smoke test), then the main run |
| `notebooks/3_evaluate.ipynb` | A100 | 1.5 | the setting chosen on validation; the test writers, once |
| `notebooks/4_owner_hand.ipynb` | A100 | 0.5 | few-shot and adapted imitation of the author's hand |
| `notebooks/5_release.ipynb` | CPU | 0 | the model and the demo on Hugging Face |
| `notebooks/6_generated_words_for_recognition.ipynb` | A100 | 1.0 | generated words as training data for the recogniser |
| `notebooks/7_ablation_no_prior.ipynb` | A100 | 3.5 | training without the printed word; quality against training |

The scripts in `scripts/` do the work; each explains itself (`python scripts/<name>.py -h`).

## Using the generator

```bash
git clone https://github.com/chingkheinganba231005/Meitei-Mayek-Diffusion-model
git clone https://github.com/chingkheinganba231005/meitei-mayek-word-recognition word_repo
git -C word_repo checkout bdee9e7
pip install -r Meitei-Mayek-Diffusion-model/requirements.txt      # PyTorch: see pytorch.org
```

```python
import sys
sys.path[:0] = ["Meitei-Mayek-Diffusion-model", "word_repo"]
import numpy as np
from PIL import Image
from mayek_diffusion.canvas import from_real
from mayek_diffusion.release import load
from mayek_diffusion.sample import generate, paper
from mayek_diffusion.writers import Printer
from mayek_words.synth import load_priors

model, config = load("release/model")          # generator.safetensors and config.json
printer = Printer(load_priors())
refs = [from_real(np.asarray(Image.open(f).convert("L")))[0] for f in ["word1.png", "word2.png"]]
words = generate(model, ["ꯃꯤꯇꯩ", "ꯃꯌꯦꯛ"], refs, printer, steps=14, sampler="dpmpp", guidance=2.0)
Image.fromarray(paper(words[0])).save("word.png")
```

`refs=None` writes in a random hand.

Tests (CPU, a few minutes):

```bash
pip install -r requirements-dev.txt gradio
WORD_REPO=word_repo pytest -q
```

## Data

- **TUMMHCD** (D. Hijam and S. Saharia, "On developing complete character set Meitei Mayek
  handwritten character database", The Visual Computer 38:525-539, 2022) is available from its
  authors at <http://agnigarh.tezu.ernet.in/~sarat/resources.html>; it is not included here.
- **Words** come from the word recognition project's word list (Meitei Wikipedia, CC BY-SA
  4.0, and FineWeb-2, ODC-By 1.0), split by word into training, validation and test words;
  the lists are not included here.
- **Real handwriting:** the 100 words of the word recognition project's `real_words/`
  (CC BY 4.0), the author's own.

## Repository

```
mayek_diffusion/  canvas, pseudo-writers, rendering, batches, model, EDM, weight averages,
                  training, sampling, evaluation, the author's hand, adaptation, release,
                  cards, the demo's helpers, generated words for the recogniser
scripts/          command-line tools used by the notebooks (each explains itself)
notebooks/        the experiments, in order
space/            the Gradio demo and the Hugging Face cards
tests/            pytest
```

## Citation

```bibtex
@misc{rajkumar2026generation,
  author       = {Rajkumar, Chingkheinganba},
  title        = {Handwritten {Meitei Mayek} word generation},
  year         = {2026},
  howpublished = {\url{https://github.com/chingkheinganba231005/Meitei-Mayek-Diffusion-model}}
}
```

## Licence

Code: MIT ([`LICENSE`](LICENSE)). The word synthesiser and its font come from the word
recognition repository (MIT; the font Noto Sans Meetei Mayek under the SIL Open Font Licence
1.1).
