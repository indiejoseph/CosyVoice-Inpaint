"""Training script for Qwen2LMInpaint

Reference: `tmp/old_train.py` (legacy CosyVoice2 fine-tune script). This script
is a minimal Trainer-based training entry point using the same datapipeline
pattern. Key changes for inpainting:

- Uses `Qwen2LMInpaint` wrapper (from `pron_inpaint.modeling`) around an
  instantiated `Qwen2LM`.
- Accepts `phoneme_token` per example (space-separated token ids) or falls back
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
from datasets import Dataset
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
from cosyvoice.utils.common import ras_sampling, IGNORE_ID
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.llm.llm import Qwen2LM

# Our inpaint implementation
from pron_inpaint.modeling import Qwen2LMInpaint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        required=True,
        help="CSV file with dataset (text, codes, optional phoneme_token)",
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
    return p.parse_args()


def convert_to_list(x: str, offset: int = 0) -> List[int]:
    return [int(i) + offset for i in x.split(" ") if i != ""]


def convert_phoneme_token(x: Optional[str], L: int) -> List[int]:
    """Return flattened 4*L phoneme token list. If missing, return zeros."""
    if x is None or str(x).strip() == "":
        return [0] * (4 * L)
    # assume x is space-separated integers
    vals = [int(i) for i in x.split(" ") if i != ""]
    if len(vals) != 4 * L:
        # If lengths mismatch, try to pad or truncate
        if len(vals) < 4 * L:
            vals = vals + [0] * (4 * L - len(vals))
        else:
            vals = vals[: 4 * L]
    return vals


class CustomDataCollatorWithPadding:
    def __init__(self, num_speech_tokens: int):
        self.num_speech_tokens = num_speech_tokens

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_tokens = [torch.LongTensor(f["text_token"]) for f in features]
        text_lens = [len(t) for t in text_tokens]
        batch = {
            "text_token": pad_sequence(
                text_tokens, batch_first=True, padding_value=151643
            ),
            "text_token_len": torch.LongTensor(text_lens),
            "speech_token": pad_sequence(
                [torch.LongTensor(f["speech_token"]) for f in features],
                batch_first=True,
                padding_value=self.num_speech_tokens,
            ),
            "speech_token_len": torch.LongTensor(
                [len(f["speech_token"]) for f in features]
            ),
            # phoneme_token is pre-flattened into space-separated ints per example
            "phoneme_token": pad_sequence(
                [torch.LongTensor(f["phoneme_token"]) for f in features],
                batch_first=True,
                padding_value=0,
            ),
        }
        return batch


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # `inputs` is already prepared by the data collator and contains our keys
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        else:
            return outputs[0]


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
        state = torch.load(llm_state, map_location="cpu")
        qwen2lm.load_state_dict(state)

    return qwen2lm


def bind_trainer_forward(inpaint_model: Qwen2LMInpaint):
    # Trainer will call model(**inputs). We expose a thin forward wrapper.
    def inpaint_trainer_forward(
        self,
        text_token: torch.LongTensor,
        text_token_len: torch.LongTensor,
        speech_token: torch.LongTensor,
        speech_token_len: torch.LongTensor,
        phoneme_token: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        device = next(self.parameters()).device
        batch = {
            "text_token": text_token.to(device),
            "text_token_len": text_token_len.to(device),
            "speech_token": speech_token.to(device),
            "speech_token_len": speech_token_len.to(device),
        }
        if phoneme_token is not None:
            batch["phoneme_token"] = phoneme_token.to(device)
        return self.forward(batch, device)

    inpaint_model.forward = inpaint_trainer_forward.__get__(inpaint_model)


def freeze_backbone_except_inpaint(inpaint_model: Qwen2LMInpaint):
    # Freeze all params in the wrapped Qwen2LM except inpaint params
    for n, p in inpaint_model.qwen2lm.named_parameters():
        p.requires_grad = False
    # Ensure inpaint module params are trainable
    for p in inpaint_model.onset_emb.parameters():
        p.requires_grad = True
    for p in inpaint_model.nucleus_emb.parameters():
        p.requires_grad = True
    for p in inpaint_model.coda_emb.parameters():
        p.requires_grad = True
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

    # load dataset CSV and filter Cantonese examples with phone annotations
    import pandas as pd
    import re

    try:
        import pycantonese
    except Exception as e:
        raise ImportError(
            "pycantonese is required to parse Jyutping annotations. Install it with `pip install pycantonese`."
        ) from e

    df = pd.read_csv(args.data)
    # keep only Cantonese samples that have phone annotations
    df = df[(df.get("lang") == "yue") & (df.get("phone").notnull())]

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_config)

    jyutping_re = re.compile(r"^[a-zA-Z]+[1-9]$")

    def parse_phone_components(phone_str: str, L: int):
        """Split phone_str by spaces and return component string lists.

        Returns (onset_l, nucleus_l, coda_l, tone_l, ok)
        - ok == False if lengths mismatch or parse failed.
        - Non-jyutping tokens (punctuation) are mapped to empty strings (""), which will
          later map to index 0 (no-phoneme).
        """
        if phone_str is None:
            return None, None, None, None, False
        parts = [p for p in phone_str.strip().split()]
        if len(parts) != L:
            return None, None, None, None, False
        onset_l, nucleus_l, coda_l, tone_l = [], [], [], []
        for tok in parts:
            if jyutping_re.match(tok):
                try:
                    parsed = pycantonese.parse_jyutping(tok)
                    syll = parsed[0]
                    try:
                        o = syll.onset
                        n = syll.nucleus
                        c = syll.coda
                        t = str(syll.tone)
                    except Exception:
                        o, n, c, t = syll[0], syll[1], syll[2], str(syll[3])
                except Exception:
                    m = re.match(r"^([a-zA-Z]+?)([1-9])$", tok)
                    if m:
                        o, n, c, t = "", m.group(1), "", m.group(2)
                    else:
                        o, n, c, t = "", "", "", ""
            else:
                # punctuation or non-jyutping -> no phoneme
                o, n, c, t = "", "", "", ""
            onset_l.append(o)
            nucleus_l.append(n)
            coda_l.append(c)
            tone_l.append(t)
        return onset_l, nucleus_l, coda_l, tone_l, True

    def tokenize_add_label(sample):
        text = tokenizer.encode(sample["text"], add_special_tokens=True)
        L = len(text)
        # speech tokens may already be a list or a space-separated string
        speech_token = sample.get("speech_tokens", [])
        if isinstance(speech_token, str):
            speech_token = convert_to_list(speech_token)
        onset_l, nucleus_l, coda_l, tone_l, ok = parse_phone_components(
            sample.get("phone", None), L
        )
        if not ok:
            return {
                "text_token": text,
                "speech_token": speech_token,
                "valid_phon": False,
            }
        return {
            "text_token": text,
            "speech_token": speech_token,
            "phoneme_onset_str": onset_l,
            "phoneme_nucleus_str": nucleus_l,
            "phoneme_coda_str": coda_l,
            "phoneme_tone_str": tone_l,
            "valid_phon": True,
        }

    ds = Dataset.from_pandas(df)
    dataset = ds.map(tokenize_add_label, remove_columns=list(ds.features), num_proc=12)
    dataset = dataset.filter(lambda ex: ex.get("valid_phon", False), num_proc=12)

    # Fixed component vocabularies (reserve index 0 for "no-phoneme" / pad)
    ONSETS = "b d g gw z p t k kw c m n ng f h s l w j".split()
    NUCLEUSES = "aa a i yu u oe e eo o m n ng".split()
    CODAS = "p t k m n ng i u".split()
    TONES = "1 2 3 4 5 6".split()

    # Build maps from string->id; index 0 is reserved for padding/no-phoneme
    onset_map = {v: i + 1 for i, v in enumerate(ONSETS)}
    nucleus_map = {v: i + 1 for i, v in enumerate(NUCLEUSES)}
    coda_map = {v: i + 1 for i, v in enumerate(CODAS)}
    tone_map = {v: i + 1 for i, v in enumerate(TONES)}

    onset_vocab_size = len(ONSETS) + 1
    nucleus_vocab_size = len(NUCLEUSES) + 1
    coda_vocab_size = len(CODAS) + 1
    tone_vocab_size = len(TONES) + 1

    def str_components_to_ids(sample):
        # maps component string lists to flattened numeric phoneme_token list
        onset_ids = [onset_map.get(s, 0) for s in sample["phoneme_onset_str"]]
        nucleus_ids = [nucleus_map.get(s, 0) for s in sample["phoneme_nucleus_str"]]
        coda_ids = [coda_map.get(s, 0) for s in sample["phoneme_coda_str"]]
        tone_ids = [tone_map.get(s, 0) for s in sample["phoneme_tone_str"]]
        flat = onset_ids + nucleus_ids + coda_ids + tone_ids
        return {"phoneme_token": flat}

    dataset = dataset.map(
        str_components_to_ids,
        remove_columns=[
            "phoneme_onset_str",
            "phoneme_nucleus_str",
            "phoneme_coda_str",
            "phoneme_tone_str",
            "valid_phon",
        ],
    )

    dataset = dataset.shuffle(seed=42)

    # split
    split = dataset.train_test_split(test_size=min(1000, len(dataset) // 10))
    train_ds, eval_ds = split["train"], split["test"]

    # Build model
    qwen2lm = build_qwen2lm(args.qwen_config, llm_state=args.llm_state)
    inpaint_model = Qwen2LMInpaint(
        qwen2lm,
        onset_vocab=onset_vocab_size,
        nucleus_vocab=nucleus_vocab_size,
        coda_vocab=coda_vocab_size,
        tone_vocab=tone_vocab_size,
        d_model=896,
        composition="concat_linear",
        gate_blend=False,
    )

    # bind forward for Trainer
    bind_trainer_forward(inpaint_model)

    # freeze backbone and leave inpaint params trainable
    freeze_backbone_except_inpaint(inpaint_model)

    # prepare Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=3,
        remove_unused_columns=False,
        label_names=["speech_token"],
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
