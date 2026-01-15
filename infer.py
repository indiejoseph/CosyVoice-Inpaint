"""
Infer script converted from infer.ipynb
Reference: /notebooks/projects/CosyVoice-Inpaint/infer.ipynb

Usage example:
  python infer.py \
    --input_text "我識得講個撚[nan2]字。" \
    --prompt_text "到咗落車嘅時候，連媽媽都走失埋。" \
    --prompt_audio tmp/ref123.wav \
    --outdir outputs
"""

import os
import sys
import uuid
import argparse
import logging

import torch
import soundfile as sf

# ensure local third_party packages are importable
SYS_PATHS = [
    "third_party/CosyVoice",
    "third_party/CosyVoice/third_party/Matcha-TTS",
]
for p in SYS_PATHS:
    if p not in sys.path:
        sys.path.append(p)

from cosyvoice.cli.cosyvoice import CosyVoice2
from transformers import AutoConfig, Qwen2ForCausalLM
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
from cosyvoice.utils.mask import make_pad_mask
from pron_inpaint.patch import patch_cosyvoice2

# Defaults are provided via CLI arguments; no module-level constants.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen2Encoder(torch.nn.Module):
    """Small wrapper around HuggingFace Qwen2ForCausalLM used in the notebook."""

    def __init__(self, config_file: str):
        super().__init__()
        config = AutoConfig.from_pretrained(config_file)
        self.model = Qwen2ForCausalLM(config)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        # used for incremental decoding in the notebook runtime
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            output_attentions=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


def setup_cosyvoice(
    device: str = None,
    qwen_pretrain_model: str = None,
    cosyvoice_dir: str = None,
    loaded_state_path: str = None,
    inpaint_weights_path: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Fallback defaults (use notebook-default paths when not provided via CLI)
    if qwen_pretrain_model is None:
        qwen_pretrain_model = (
            "/notebooks/projects/tts2025/results/cosyvice2-20250907/blank"
        )
    if cosyvoice_dir is None:
        cosyvoice_dir = (
            "/notebooks/projects/CosyVoice/pretrained_models/CosyVoice2-0.5B"
        )
    if loaded_state_path is None:
        loaded_state_path = (
            "/notebooks/projects/tts2025/results/cosyvice2-20250907/model_acc_0.1813.pt"
        )
    if inpaint_weights_path is None:
        inpaint_weights_path = (
            "/notebooks/projects/CosyVoice-Inpaint/results/inpaint/pytorch_model.bin"
        )

    logger.info("Loading CosyVoice from %s", cosyvoice_dir)
    cosyvoice = CosyVoice2(cosyvoice_dir, fp16=True, load_trt=True)

    cosyvoice.model.llm.to(device)
    cosyvoice.model.flow.to(device)

    # Patch tokenizer like the notebook
    cosyvoice.frontend.tokenizer = get_qwen_tokenizer(
        token_path=qwen_pretrain_model, skip_special_tokens=True, version="cosyvoice2"
    )

    # Replace llm.llm with the encoder wrapper
    logger.info("Attaching Qwen2Encoder")
    cosyvoice.model.llm.llm = Qwen2Encoder(qwen_pretrain_model).to(device)

    # load pretrained qwen weights if available (not strictly necessary for patched flow)
    if os.path.exists(loaded_state_path):
        logger.info("Loading LLM state dict: %s", loaded_state_path)
        try:
            loaded_state_dict = torch.load(loaded_state_path, weights_only=True)
        except TypeError:
            loaded_state_dict = torch.load(loaded_state_path)
        try:
            cosyvoice.model.llm.load_state_dict(loaded_state_dict)
        except Exception as e:
            logger.warning("Failed to strictly load LLM weights: %s", e)

    # Apply pron_inpaint patch (if available) and load fine-tuned inpaint weights
    logger.info("Applying pron_inpaint.patch.patch_cosyvoice2")
    patch_cosyvoice2(cosyvoice)

    if os.path.exists(inpaint_weights_path):
        logger.info("Loading inpaint fine-tuned weights: %s", inpaint_weights_path)
        inpaint_state_dict = torch.load(inpaint_weights_path, map_location=device)
        # Notebook: phone_emb was accidentally sized 49 -> truncate to 46
        if "phone_emb.weight" in inpaint_state_dict:
            w = inpaint_state_dict["phone_emb.weight"]
            if w.size(0) > 46:
                logger.info("Truncating phone_emb.weight from %d -> 46", w.size(0))
                inpaint_state_dict["phone_emb.weight"] = w[:46, :]
        cosyvoice.model.llm.load_state_dict(inpaint_state_dict, strict=False)

    cosyvoice.model.llm.to(device)
    cosyvoice.device = device

    return cosyvoice


def run_inference(cosyvoice, input_text, prompt_text, prompt_audio, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    text = cosyvoice.frontend.text_normalize(
        input_text, split=False, text_frontend=True
    )

    model_inputs = cosyvoice.frontend.frontend_zero_shot(
        text, prompt_text, prompt_audio, cosyvoice.sample_rate, zero_shot_spk_id=""
    )
    logger.info("Model inputs keys: %s", list(model_inputs.keys()))

    # Show token decode for debugging (like in the notebook)
    try:
        decoded = cosyvoice.frontend.frontend.tokenizer.tokenizer.decode(
            model_inputs["text"][0], skip_special_tokens=False
        )
        logger.info("Decoded tokenizer sample: %s", decoded)
    except Exception as e:
        logger.debug("Failed to decode tokens: %s", e)

    for idx, o in enumerate(
        cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_audio)
    ):
        wav = o["tts_speech"].cpu().numpy().squeeze()
        rate = 24000
        fname = os.path.join(outdir, f"tts_{uuid.uuid4().hex[:8]}_{idx}.wav")
        sf.write(fname, wav, rate)
        logger.info("Wrote %s (len=%d samples)", fname, len(wav))

    return outdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_text",
        default="我識得講個撚[nan2]字。",
        help="Input text with inpaint tokens",
    )
    parser.add_argument(
        "--prompt_text",
        default="到咗落車嘅時候，連媽媽都走失埋。",
        help="Prompt/reference text",
    )
    parser.add_argument(
        "--prompt_audio",
        default="tmp/ref123.wav",
        help="Path to prompt/reference audio",
    )
    parser.add_argument("--outdir", default="outputs", help="Output directory for wavs")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    parser.add_argument(
        "--qwen_pretrain_model",
        default="/notebooks/projects/tts2025/results/cosyvice2-20250907/blank",
        help="Path to the qwen pretrain model token/config",
    )
    parser.add_argument(
        "--cosyvoice_dir",
        default="/notebooks/projects/CosyVoice/pretrained_models/CosyVoice2-0.5B",
        help="Path to CosyVoice pretrained model directory",
    )
    parser.add_argument(
        "--loaded_state",
        default="/notebooks/projects/tts2025/results/cosyvice2-20250907/model_acc_0.1813.pt",
        help="Path to the pretrained LLM state file",
    )
    parser.add_argument(
        "--inpaint_weights",
        default="/notebooks/projects/CosyVoice-Inpaint/results/inpaint/pytorch_model.bin",
        help="Path to the inpaint fine-tuned weights",
    )
    args = parser.parse_args()

    cosyvoice = setup_cosyvoice(
        device=args.device,
        qwen_pretrain_model=args.qwen_pretrain_model,
        cosyvoice_dir=args.cosyvoice_dir,
        loaded_state_path=args.loaded_state,
        inpaint_weights_path=args.inpaint_weights,
    )

    run_inference(
        cosyvoice, args.input_text, args.prompt_text, args.prompt_audio, args.outdir
    )


if __name__ == "__main__":
    main()
