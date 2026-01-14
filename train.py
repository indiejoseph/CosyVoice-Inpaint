"""Training script for Qwen2LMInpaint

Reference: `tmp/old_train.py` (legacy CosyVoice2 fine-tune script). This script
is a minimal Trainer-based training entry point using the same datapipeline
pattern. Key changes for inpainting:

- Uses `Qwen2LMInpaint` wrapper (from `pron_inpaint.modeling`) around an
  instantiated `Qwen2LM`.
- Accepts `phone_token` per example (space-separated token ids) or falls back
  to zeros (no inpainting).
- Provides a forward wrapper `inpaint_trainer_forward` bound to the inpaint
  model so that `Trainer` can call `model(**inputs)`.

Usage example:
    python train.py --data dataset.csv --output_dir results/inpaint1 --epochs 3

"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen2ForCausalLM,
    Trainer,
    TrainingArguments,
)

# Add local cosyvoice and pron_inpaint to path
sys.path.append("third_party/CosyVoice")
from cosyvoice.utils.common import ras_sampling
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.llm.llm import Qwen2LM

# Our inpaint implementation
from pron_inpaint.modeling import Qwen2LMInpaint
from pron_inpaint.frontend_wrapper import InpaintFrontendWrapper
from pron_inpaint.tokenizer import JyutpingTokenizer, TONE_OFFSET


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        required=True,
        help="CSV file with dataset (text, codes, optional phone_token)",
    )
    p.add_argument(
        "--qwen_config",
        required=True,
        help="Path to Qwen2 config folder or file for Qwen2ForCausalLM",
    )
    p.add_argument("--llm_state", default=None, help="Optional path to llm.pt weights")
    p.add_argument("--output_dir", default="results/inpaint")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # phoneme mixing: keep probability for phoneme tokens (3:1 text:phoneme => 0.25)
    p.add_argument(
        "--phoneme_keep_prob",
        type=float,
        default=0.25,
        help="Probability to keep a phoneme token (per token)",
    )
    p.add_argument(
        "--phoneme_seed",
        type=int,
        default=42,
        help="Deterministic seed for phoneme masking",
    )
    p.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Reporting backend for Trainer (e.g., 'none' or 'wandb')",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of warmup steps for the optimizer (passed to Transformers TrainingArguments)",
    )
    p.add_argument(
        "--dataset_factor",
        type=int,
        default=1,
        help=(
            "Factor to repeat the dataset (e.g., 4 to expand the dataset 4x). "
            "This simply repeats dataset entries; no processing logic is changed."
        ),
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 training (uses TrainingArguments.bf16). Requires hardware support.",
    )
    return p.parse_args()


def convert_to_list(x: str, offset: int = 0) -> List[int]:
    return [int(i) + offset for i in x.split(" ") if i != ""]


class CustomDataCollatorWithPadding:
    def __init__(self, num_speech_tokens: int):
        self.num_speech_tokens = num_speech_tokens

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_tokens = [torch.LongTensor(f["text_token"]) for f in features]
        text_lens = [len(t) for t in text_tokens]

        # Speech padding
        speech_tokens = [torch.LongTensor(f["speech_token"]) for f in features]
        speech_lens = [len(s) for s in speech_tokens]

        # phone_token is pre-flattened into space-separated ints per example
        # Use simple pad_sequence on the flattened vectors and expose phone_token_len
        phone_tensors = [torch.LongTensor(f["phone_token"]) for f in features]
        phone_lens = [len(f["phone_token"]) // 4 for f in features]

        batch = {
            "text_token": pad_sequence(
                text_tokens, batch_first=True, padding_value=151643
            ),
            "text_token_len": torch.LongTensor(text_lens),
            "speech_token": pad_sequence(
                speech_tokens, batch_first=True, padding_value=self.num_speech_tokens
            ),
            "speech_token_len": torch.LongTensor(speech_lens),
            "phone_token": pad_sequence(
                phone_tensors, batch_first=True, padding_value=0
            ),
            "phone_token_len": torch.LongTensor(phone_lens),
        }
        return batch


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(inputs, device=self.args.device)

        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        else:
            # Fallback for tuple outputs
            loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            return (loss, outputs) if return_outputs else loss


def build_qwen2lm(qwen_config_path: str, llm_state: Optional[str] = None):
    # Build Qwen2 encoder wrapper
    class Qwen2Encoder(torch.nn.Module):
        def __init__(self, config_path: str):
            super().__init__()
            config = AutoConfig.from_pretrained(config_path)
            self.model = Qwen2ForCausalLM(config)

        def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
            T = xs.size(1)
            masks = ~make_pad_mask(xs_lens, T)
            outs = self.model(
                inputs_embeds=xs,
                attention_mask=masks,
                output_hidden_states=True,
                return_dict=True,
            )
            return outs.hidden_states[-1], masks.unsqueeze(1)

        def forward_one_step(self, xs, masks, cache=None):
            input_masks = masks[:, -1, :]
            outs = self.model(
                inputs_embeds=xs,
                attention_mask=input_masks,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
                past_key_values=cache,
            )
            xs = outs.hidden_states[-1]
            new_cache = outs.past_key_values
            return xs, new_cache

    encoder = Qwen2Encoder(qwen_config_path)

    qwen2lm = Qwen2LM(
        llm_input_size=896,
        llm_output_size=896,
        speech_token_size=6561,
        llm=encoder,
        sampling=ras_sampling,
        length_normalized_loss=True,
    )

    # optionally load llm state
    if llm_state is not None:
        state = torch.load(llm_state, map_location="cpu", weights_only=False)
        qwen2lm.load_state_dict(state)

    return qwen2lm


def freeze_backbone_except_inpaint(inpaint_model: Qwen2LMInpaint):
    # Freeze all params in the wrapped Qwen2LM except inpaint params
    for n, p in inpaint_model.qwen2lm.named_parameters():
        p.requires_grad = False
    # Ensure inpaint module params are trainable
    # unified phone embedding
    if hasattr(inpaint_model, "phone_emb"):
        for p in inpaint_model.phone_emb.parameters():
            p.requires_grad = True
    else:
        # fallback to old per-component attributes if present
        if hasattr(inpaint_model, "onset_emb"):
            for p in inpaint_model.onset_emb.parameters():
                p.requires_grad = True
        if hasattr(inpaint_model, "nucleus_emb"):
            for p in inpaint_model.nucleus_emb.parameters():
                p.requires_grad = True
        if hasattr(inpaint_model, "coda_emb"):
            for p in inpaint_model.coda_emb.parameters():
                p.requires_grad = True
        if hasattr(inpaint_model, "tone_emb"):
            for p in inpaint_model.tone_emb.parameters():
                p.requires_grad = True
    if hasattr(inpaint_model, "composer") and inpaint_model.composer is not None:
        for p in inpaint_model.composer.parameters():
            p.requires_grad = True
    if (
        hasattr(inpaint_model, "gate_blend")
        and inpaint_model.gate_blend
        and hasattr(inpaint_model, "gate")
    ):
        for p in inpaint_model.gate.parameters():
            p.requires_grad = True


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset: if a CSV file path is provided, use pandas to read and convert to a Dataset;
    # otherwise prefer HuggingFace `load_dataset` and fall back to `load_from_disk`.
    if isinstance(args.data, str) and args.data.lower().endswith(".csv"):
        try:
            import pandas as pd  # optional dependency
        except Exception:
            raise RuntimeError(
                "pandas is required to load a CSV dataset; please install it (pip install pandas)"
            )
        df = pd.read_csv(args.data)
        ds = Dataset.from_pandas(df)
    else:
        try:
            ds = load_dataset(args.data)
        except Exception:
            ds = load_from_disk(args.data)

    # instantiate a JyutpingTokenizer for parsing phoneme annotations
    jyutoken = JyutpingTokenizer()

    # Use the reusable tokenize_add_label implementation from pron_inpaint.utils
    from pron_inpaint.utils import tokenize_add_label

    # First pass: insert bracketed Jyutping tokens into raw text according to `--phoneme_keep_prob`
    dataset = ds.map(
        tokenize_add_label,
        fn_kwargs={
            "insert_prob": args.phoneme_keep_prob,
            "seed": args.phoneme_seed,
        },
        remove_columns=list(ds.features) if hasattr(ds, "features") else None,
        num_proc=12,
    )

    # keep only examples that had a parsable phone annotation
    dataset = dataset.filter(lambda ex: ex.get("valid_phon", False), num_proc=12)

    # Second pass (single-process): use the InpaintFrontendWrapper to normalize the
    # text and extract `text_token` and `phone_token` buffers aligned to text tokens.
    def _apply_frontend(ex, tokenizer_path=None):
        # Cache frontend wrapper on the function object to avoid instantiating it for
        # every example (expensive). This is safe when running with num_proc=1.
        if not hasattr(_apply_frontend, "_fw") or _apply_frontend._fw is None:
            _apply_frontend._fw = InpaintFrontendWrapper(tokenizer_path=tokenizer_path)
        fw = _apply_frontend._fw

        text_with = ex.get("text_with_jyutping", ex.get("text", ""))
        # Normalize without splitting to keep one segment
        norm_text = fw.text_normalize(text_with, split=False, text_frontend=True)
        text_token, _ = fw._extract_text_token(norm_text)
        phone_token, _ = fw._extract_phone_token(norm_text, text_token)
        text_tokens_list = text_token[0].tolist() if text_token.numel() > 0 else []
        phone_tokens_list = phone_token[0].tolist() if phone_token.numel() > 0 else []

        return {
            "text_token": text_tokens_list,
            "speech_token": ex.get("speech_token", ex.get("speech_tokens", [])),
            "phone_token": phone_tokens_list,
            "valid_phon": True if len(phone_tokens_list) > 0 else False,
        }

    dataset = dataset.map(
        _apply_frontend,
        fn_kwargs={"tokenizer_path": args.qwen_config},
        # keep only the columns we need downstream
        remove_columns=[
            c
            for c in dataset.column_names
            if c not in ("text_token", "speech_token", "phone_token", "valid_phon")
        ],
        num_proc=1,
    )

    # Final filter: ensure we actually produced phoneme buffers aligned to the text
    dataset = dataset.filter(lambda ex: ex.get("valid_phon", False), num_proc=12)

    # Optionally expand the dataset by repeating entries `dataset_factor` times. This is a
    # straightforward way to increase the effective coverage of inserted phonemes (e.g., with
    # insert_prob=0.25 and dataset_factor=4, the expanded dataset will contain ~100% annotated examples).
    if (
        hasattr(args, "dataset_factor")
        and args.dataset_factor
        and args.dataset_factor > 1
    ):
        from datasets import concatenate_datasets

        dataset = concatenate_datasets([dataset] * int(args.dataset_factor))
        # shuffle again to mix the duplicates
        dataset = dataset.shuffle(seed=42)

    # Use vocab sizes provided by the JyutpingTokenizer and compute a single unified vocab_size
    vocab_size = jyutoken.vocab_size()

    # drop helper columns and keep phone_token (already numeric) from tokenize_add_label
    dataset = dataset.map(
        lambda ex: {k: ex[k] for k in ex if k not in ("valid_phon",)},
        remove_columns=["valid_phon"],
        num_proc=12,
    )

    dataset = dataset.shuffle(seed=42)

    # split
    split = dataset.train_test_split(test_size=min(1000, len(dataset) // 10))
    train_ds, eval_ds = split["train"], split["test"]

    # Build model
    qwen2lm = build_qwen2lm(args.qwen_config, llm_state=args.llm_state)
    inpaint_model = Qwen2LMInpaint(
        qwen2lm,
        vocab_size=vocab_size,
        tone_offset=TONE_OFFSET,
        composition="concat_linear",
    )
    # Initialize phone embedding weights from unified embedding
    inpaint_model.init_component_from_text_embed()

    # freeze backbone and leave inpaint params trainable
    freeze_backbone_except_inpaint(inpaint_model)

    # prepare Trainer
    # warn if bf16 requested but hardware might not support it
    if args.bf16 and not torch.cuda.is_available():
        print(
            "Warning: --bf16 requested but CUDA/accelerator not detected; training may fail if hardware does not support bfloat16."
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        bf16=args.bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=3,
        remove_unused_columns=False,
        label_names=["speech_token"],
        save_safetensors=False,  # Must be False due to tied embedding in Qwen2LM
        report_to=args.report_to,
    )

    data_collator = CustomDataCollatorWithPadding(num_speech_tokens=6561)

    trainer = CustomTrainer(
        model=inpaint_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
