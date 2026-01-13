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
Qwen2LMInpaint(Qwen2LM)
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

* Gated blending of text and phoneme embeddings
* Learnable mixing coefficients

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
