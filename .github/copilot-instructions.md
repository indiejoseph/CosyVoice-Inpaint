# Copilot / AI Agent Instructions — CosyVoice-Inpaint

Short summary
- Purpose: add pronunciation inpainting to CosyVoice by composing phonological component embeddings (onset, nucleus, coda, tone) and injecting composed phoneme embeddings into a frozen CosyVoice LLM (Qwen2LM) via a `Qwen2LMInpaint` subclass.
- Keep CosyVoice upstream unchanged — it is a git submodule at `third_party/CosyVoice`.

What an agent should know immediately (big-picture)
- Architecture: The LLM (CosyVoice / Qwen2LM) is kept fully **frozen**. Trainable parts live in separate modules: small embedding tables (onset/nucleus/coda/tone), a composition network (MLP/linear/gated fusion), optional gating/projection layers, and optional regularizers.
- Data flow: During training/inference, replace selected text spans with composed phoneme embeddings; pass the full embedding sequence unchanged through the frozen transformer; compute acoustic reconstruction loss backpropagated only to phoneme modules.
- Key design constraints: keep output embedding dim == LLM d_model; avoid modifying the submodule code. Prefer subclassing or adding a new top-level package that imports CosyVoice.

Critical files & places to look
- Project README: `README.md` (root) — concise design, training assumptions and targeted risks.
- CosyVoice runtime and examples: `third_party/CosyVoice/README.md`, `third_party/CosyVoice/example.py`, `third_party/CosyVoice/vllm_example.py`, `third_party/CosyVoice/webui.py`, and `third_party/CosyVoice/examples/libritts`.
- Docker & runtime: `third_party/CosyVoice/runtime` and `third_party/CosyVoice/runtime/triton_trtllm` for deployment/acceleration patterns.

Typical developer workflows (concrete commands)
- Clone (submodules required):
  - `git clone --recursive https://github.com/your/repo.git` or `git submodule update --init --recursive`
- Create environment and install (CosyVoice uses Conda + pip):
  - `conda create -n cosyvoice -y python=3.10 && conda activate cosyvoice`
  - `pip install -r third_party/CosyVoice/requirements.txt`
- Download models (modelscope or Hugging Face): use `snapshot_download` from `modelscope` or `huggingface_hub` as shown in `third_party/CosyVoice/README.md`.
- Quick run / debug:
  - `python third_party/CosyVoice/example.py` (basic usage)
  - `python third_party/CosyVoice/vllm_example.py` (vLLM usage — note strict vLLM versions)
  - `python third_party/CosyVoice/webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M`
- Docker-based server (runtime/python):
  - `cd third_party/CosyVoice/runtime/python && docker build -t cosyvoice:v1.0 .`
  - Run GRPC or FastAPI containers as shown in the CosyVoice README.
- vLLM versions: use vLLM 0.9.0 (legacy) or >=0.11.0 (V1 engine); other versions are not tested.

Project-specific patterns & constraints (do not generalize!)
- Submodule boundary: do NOT modify `third_party/CosyVoice` directly in PRs. If a change is required upstream, either 1) make a fork and change the submodule reference, or 2) implement behavior by subclassing/wrapping CosyVoice code in this repository.
- Frozen LLM + small trainable modules: the intended approach is to inject embeddings rather than modifying tokenizer or full LLM weights.
- Phoneme insertion: multi-syllable words should insert multiple phoneme embeddings in-order; timing/sequence must be preserved.
- Regularizer example: optionally add an auxiliary loss like `||LLM(E_phoneme) - LLM(E_text_equivalent)||` to keep composed embeddings on-manifold.

Code & PR guidance for agents
- Where to add code: create a top-level package (e.g., `inpaint/` or `src/inpaint/`) that imports CosyVoice from `third_party/CosyVoice` and implements `Qwen2LMInpaint` and phoneme modules.
- Small, self-contained changes: prefer adding new modules, tests, and examples (small demo scripts) rather than editing submodule files.
- Tests & validation: there are no formal unit tests in this repo; validate locally using `example.py`, small synthetic inputs, and the CosyVoice `webui.py` for manual inspection.
- Performance/dev tips: if GPU or inference speed needed, follow the `runtime/triton_trtllm` path or Docker recipes; be mindful of vLLM compatibility and CUDA/TensorRT dependencies.

Examples to copy/paste (use these exact commands for reproducibility)
- Submodules: `git submodule update --init --recursive`
- Conda + requirements: `conda create -n cosyvoice -y python=3.10 && conda activate cosyvoice && pip install -r third_party/CosyVoice/requirements.txt`
- Run demo: `python third_party/CosyVoice/example.py`
- Download model (modelscope):
  ```py
  from modelscope import snapshot_download
  snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
  ```

What *not* to do
- Don't change or patch `third_party/CosyVoice` directly in a way that breaks the ability to pull upstream updates.
- Don't assume vLLM compatibility across versions — test with the versions specified in `third_party/CosyVoice/README.md`.

If something is unclear or missing
- Check `README.md` (root) for algorithmic assumptions and `third_party/CosyVoice/README.md` for build/runtime details.
- Ask: Where should inpaint training scripts live? (We recommend `examples/inpaint_train.py` and small CI/validation scripts.)

Contact
- For changes that touch CosyVoice behavior, prefer opening an issue in the upstream `FunAudioLLM/CosyVoice` repository and reference the PR here.

---
Please review these instructions and tell me any missing examples or workflows you'd like added (CI, test scripts, or preferred file locations).