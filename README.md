# CosyVoice Pronunciation Inpainting

This project explores **pronunciation inpainting for CosyVoice** by introducing **compositional phoneme embeddings** that can be injected into a frozen LLM. The goal is to achieve fine-grained, localized pronunciation control (e.g. correcting a single word or syllable) **without fine-tuning the CosyVoice language model or modifying its tokenizer**.

---

## Motivation

Most LLM-based TTS systems, including **CosyVoice**, consume raw text tokens and implicitly learn pronunciation from data. While effective, this makes it difficult to:

* Correct the pronunciation of a specific word
* Control accent, tone, or stress
* Perform localized pronunciation edits without affecting the whole utterance

Pronunciation inpainting addresses this by allowing **explicit, local control** over pronunciation while keeping the rest of the text unchanged.

---

## Core Idea

Instead of introducing new discrete pronunciation tokens or fine-tuning the LLM, this project proposes:

* Representing pronunciation as **phonological components**:

  * Onset
  * Nucleus
  * Coda
  * Tone
* Learning **continuous embeddings** for each component
* Composing them into a single **phoneme embedding**
* Injecting that embedding directly into the CosyVoice LLM input sequence

The CosyVoice LLM remains **fully frozen**. Only the phoneme embedding and composition modules are trained.

---

## Phoneme Composition

### Phonological Components

Each component has its own embedding table:

* Onset embedding
* Nucleus embedding
* Coda embedding
* Tone embedding

These components are linguistically grounded and allow **compositional generalization** to unseen syllables.

### Composition Function

Given `(onset, nucleus, coda, tone)`, a trainable composition network produces:

```
E_phoneme ∈ R^{d_model}
```

Possible implementations include:

* Linear projection over concatenated embeddings
* Small MLP
* Gated additive fusion (recommended)

The output dimension matches the CosyVoice LLM embedding size.

---

## Integration with CosyVoice

### Git Submodule

CosyVoice is included as a git submodule to avoid upstream modification:

```
third_party/
  └── CosyVoice/
```

---

### Qwen2LMInpaint

CosyVoice uses `Qwen2LM` as its core language model. This project introduces:

```
Qwen2LMInpaint(Qwen2LM) — now accepts a single `vocab_size` arg (total phoneme vocab size) and uses a unified `phone_emb`.
```

This subclass:

* Reuses all original Qwen2LM parameters (frozen)
* Adds trainable phoneme embedding and composition modules
* Overrides input embedding construction to support pronunciation inpainting

No changes are made to the original CosyVoice code.

---

## Pronunciation Inpainting

### Input Annotation Format

Pronunciation inpainting is enabled via **inline phoneme annotations** embedded directly in the input text.

#### Example

```
Text input:        你好呀！
Phoneme annotation: 你好呀[aa3]！
```

Here, `aa3` is the **Jyutping** pronunciation of the character `呀`.

---

### Parsing Jyutping Annotations

This project uses **pycantonese** to parse Jyutping annotations:

```python
import pycantonese
pycantonese.parse_jyutping("aa3")
```

The parsed result is decomposed into phonological components:

* **Onset**: `""`
* **Nucleus**: `"aa"`
* **Coda**: `""`
* **Tone**: `"3"`

These components are then mapped to their corresponding embedding IDs.

---

### Phoneme Token Processor

A dedicated **phoneme processor** is responsible for:

1. Scanning input text for inline phoneme annotations (e.g. `[aa3]`)
2. Removing annotation markers from the visible text
3. Parsing Jyutping into `(onset, nucleus, coda, tone)` using `pycantonese`
4. Converting parsed components into phoneme embedding indices
5. Emitting:

   * Cleaned text tokens
   * Pronunciation inpainting metadata describing replacement spans

This processor operates **before** tokenization and LLM embedding lookup.

### Phoneme token layout (interleaved per-token)

We use an *interleaved, per-text-token* flattened phoneme format to align phonemes to token positions.

- `phoneme_token` is a flat vector with shape `(B, 4 * L)` where `L` is the padded text token length and each text token owns 4 consecutive slots ordered as:

  `[onset_0, nucleus_0, coda_0, tone_0, onset_1, nucleus_1, coda_1, tone_1, ...]`

- Two common construction patterns are supported:
  - If your phoneme processor already pads to text length, pass `phoneme_token` shaped `(B, 4*L)` and omit `phoneme_token_len`.
  - If your phoneme processor returns per-sample flattened lists (length `4*P` where `P <= L` is the number of phoneme tokens), pass that flattened vector *and* `phoneme_token_len` — a `(B,)` tensor of integers containing `P` (the number of phoneme tokens for each sample). The model will place the first `4*P` entries into a `(B, 4, L)` buffer and leave remaining positions as zeros.

**Important:** `phoneme_token_len` refers to the *number of phoneme tokens (P)*, not the flattened slot count `4*P` nor the padded flattened length `4*L`.

---

### Embedding Replacement

During embedding construction:

* Standard text spans use the normal text embedding lookup
* Annotated spans are replaced with **composed phoneme embeddings** derived from the parsed Jyutping

This enables precise, localized pronunciation control while leaving surrounding text untouched.

---

### Input-Level Injection

During training and inference:

* Normal text tokens use the standard text embedding lookup
* Selected spans are replaced with **composed phoneme embeddings**

The resulting embedding sequence is passed unchanged through the frozen transformer layers.

### Multi-Syllable Words

For multi-syllable words, multiple phoneme embeddings are inserted sequentially, preserving syllable order and timing.

---

## Training

### Frozen LLM

All CosyVoice LLM parameters are frozen. Trainable components include:

* Phoneme component embeddings
* Phoneme composition network
* Optional gating or projection layers

### Loss Function

Training uses CosyVoice’s existing acoustic reconstruction loss (e.g. diffusion or flow-based). Gradients flow **only** into the phoneme-related modules.

### Optional Regularization

An auxiliary regularization loss may be added to keep phoneme embeddings close to the text embedding manifold:

```
|| LLM(E_phoneme) − LLM(E_text_equivalent) ||
```

---

### Quickstart: training example ✅

To run a quick training job (expects a CSV with columns `text`, `speech_tokens`, and optional `phone`):

```bash
python train.py --data dataset.csv \
  --qwen_config path/to/qwen/config \
  --output_dir results/inpaint \
  --epochs 3 \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-4 \
  --phoneme_keep_prob 0.25 \
  --phoneme_seed 42 \
  --report_to none
```

**Notes:**

- **`--phoneme_keep_prob`** controls the fraction of inline Jyutping annotations inserted into the text (default: `0.25`). ✅
- **`--dataset_factor`** repeats dataset entries to expand the dataset by that factor (default: `1`). Example: `--phoneme_keep_prob 0.25 --dataset_factor 4` will result in approximately 100% of training examples containing inline annotations. 🔁
- The training pipeline uses `tokenize_add_label` (to insert bracketed Jyutping) and `InpaintFrontendWrapper` (to normalize text and extract `text_token` and `phoneme_token`). 🔧
- If your dataset is a CSV, **`pandas`** is required by `train.py` to load it.
- Make sure the CosyVoice submodule is initialized before running: `git submodule update --init --recursive`. ⚠️

---

---

## Advantages

* No LLM fine-tuning required
* No tokenizer or vocabulary changes
* Linguistically grounded pronunciation control
* Efficient training with a small number of parameters
* Compatible with existing CosyVoice checkpoints

---

## Risks and Mitigations

### Embedding Distribution Mismatch

**Risk:** Composed phoneme embeddings may lie off the LLM embedding manifold.

**Mitigations:**

* Initialize from averaged text embeddings
* Apply L2 regularization
* Use low-rank projection into LLM space

### Over-Constraint

**Risk:** Excessive pronunciation control may reduce natural coarticulation.

**Mitigations:**

* Learnable mixing coefficients (optional)

---

## Installation ✅

Clone the repository and initialize nested submodules, then install Python dependencies in the order shown (CosyVoice requirements first):

```bash
# Clone with submodules (recommended):
git clone --recursive <repo-url>

# If already cloned, sync and initialize all submodules (including nested ones like Matcha-TTS):
git submodule sync --recursive
git submodule update --init --recursive

# Install requirements (install CosyVoice's requirements first):
pip install -r third_party/CosyVoice/requirements.txt
pip install -r requirements.txt
```

> Note: The CosyVoice submodule contains its own nested `third_party` dependencies (for example `Matcha-TTS`); using `--recursive` ensures these are fetched and checked out at the pinned commits.

---

## Project Status

This repository currently contains:

* Project scaffolding
* Design and implementation plan

Model implementation and experiments are work in progress.

---

## License

This project is released under the **Apache License 2.0**.

CosyVoice is included as a git submodule and remains licensed under its original license. Please refer to the CosyVoice repository for details.

```
Copyright 2026

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
