"""Pronunciation inpainting modeling helpers

Implements Qwen2LMInpaint which wraps an existing Qwen2LM (or CosyVoice3LM)
and provides:

- small phonological component embedding tables (onset/nucleus/coda/tone)
- a composition network to build `E_phoneme` in the same d_model as the LLM
- a conservative, per-token replacement strategy: when phoneme components are
  provided for a token position, the corresponding text token embedding is
  replaced with the composed phoneme embedding (optionally gated/blended)

Notes / API conventions (deliberately simple to start):
- Batch-level expected keys (optional):
  - `phone_token` with shape `(B, 4 * L)` where `L` is the `text_token` length. The layout is interleaved per token:
    `[onset_0, nucleus_0, coda_0, tone_0, onset_1, nucleus_1, coda_1, tone_1, ...]`.
    The implementation will rearrange this into four `(B, L)` component tensors `(onset, nucleus, coda, tone)`. An index of `0` means "no phoneme".
  - If this key is absent, behavior falls back to standard Qwen2LM forward.
- Multi-syllable insertion (multiple phoneme embeddings per text token)
  is *not* supported by this initial implementation. It is documented and may
  be extended later.

"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Generator

sys.path.append("../third_party/CosyVoice")

from cosyvoice.utils.file_utils import logging

if TYPE_CHECKING:
    from cosyvoice.llm.llm import Qwen2LM


class Qwen2LMInpaint(nn.Module):
    """Wraps a `Qwen2LM` instance and provides phoneme inpainting utilities.

    Args:
        qwen2lm: an instantiated Qwen2LM (or CosyVoice3LM) from third_party/CosyVoice
        vocab_size: total phoneme vocabulary size (includes pad=0 and covers all components)
        composition: one of {'concat_linear', 'sum'} (default 'concat_linear')
    """

    def __init__(
        self,
        qwen2lm: "Qwen2LM",
        vocab_size: int,
        tone_offset: int,
        composition: str = "concat_linear",
        num_tones=7,  # 6 for Cantonese + 1 for pad / no tone
        tone_weight=0.3,
    ):
        """Note: `vocab_size` must be the total phoneme vocabulary size including pad idx 0.
        For example: vocab_size = (n_onset + n_nucleus + n_coda + n_tone + 1)
        """
        super().__init__()
        self.qwen2lm = qwen2lm
        self.d_model = qwen2lm.llm_input_size
        assert composition in ("concat_linear", "sum")
        self.composition = composition
        self.tone_weight = tone_weight
        self.tone_offset = tone_offset

        # unified phoneme embedding table: index 0 is reserved as "no phoneme" / padding
        total_vocab = vocab_size
        self.phone_emb = nn.Embedding(total_vocab, self.d_model, padding_idx=0)
        self.composer = torch.nn.Identity()
        if composition == "concat_linear":
            self.composer = nn.Sequential(
                nn.Linear(
                    3 * self.d_model, self.d_model
                ),  # onset+nucleus+coda concatenated, tone as residual
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
        self.num_tones = num_tones
        self.tone_classifier = nn.Linear(self.d_model, self.num_tones)

    def compose_phoneme(
        self, phoneme_flat: torch.LongTensor, phone_token_len: torch.LongTensor = None
    ) -> tuple:
        """Compose phoneme embeddings from a flattened phoneme id tensor.

        Args:
            phoneme_flat: LongTensor of shape (B, 4*L) where the layout is interleaved per token:
                [onset_0, nucleus_0, coda_0, tone_0, onset_1, nucleus_1, ...]
            phone_token_len: optional (B,) unpadded per-sample text token lengths

        Returns:
            tuple((B, L, d_model) composed phoneme embeddings, (B, L) boolean mask indicating which token positions contain any phoneme component)

        Behavior:
            If `phone_token_len` is provided, the method will reconstruct a
            (B, 4, Lmax) per-component buffer by slicing each sample's first
            `4 * Li` entries and placing them into the buffer. This supports
            batched inputs where `phoneme_flat` was padded via `pad_sequence()`
            (i.e., tail padding on the flattened vector).
        """
        if phoneme_flat.dim() != 2:
            raise ValueError("phoneme_flat must be shape (B, 4*L)")
        B, PT_len = phoneme_flat.shape
        if PT_len % 4 != 0:
            raise ValueError("phoneme_flat second dimension must be divisible by 4")
        L = PT_len // 4

        # Reconstruct per-component tensor (expect interleaved per-token format)
        if phone_token_len is not None:
            if phone_token_len.dim() != 1 or phone_token_len.numel() != B:
                raise ValueError(
                    "phone_token_len must be shape (B,) matching batch size"
                )
            Lmax = L
            pf4 = torch.zeros(
                (B, 4, Lmax), dtype=phoneme_flat.dtype, device=phoneme_flat.device
            )
            for i in range(B):
                Li = int(phone_token_len[i].item())
                if Li < 0 or Li > Lmax:
                    raise ValueError(
                        "phone_token_len values must be between 0 and Lmax"
                    )
                if 4 * Li > PT_len:
                    raise ValueError(
                        "phoneme_flat too short for declared phone_token_len"
                    )
                if Li > 0:
                    seg = phoneme_flat[i, : 4 * Li]  # shape (4*Li,)
                    seg = seg.view(Li, 4)  # (Li,4) -> per-token [on,nuc,co,to]
                    pf4[i, :, :Li] = seg.t()  # transpose to (4,Li)
            pf_flat = pf4.view(B, 4 * Lmax)
            emb = self.phone_emb(pf_flat)
        else:
            # assume phoneme_flat is arranged interleaved per token:
            # [onset_0, nucleus_0, coda_0, tone_0, onset_1, ...]
            pf4 = phoneme_flat.view(B, L, 4).permute(0, 2, 1)  # (B,4,L)
            pf_flat = pf4.reshape(B, 4 * L)
            emb = self.phone_emb(pf_flat)

        # reshape to (B, 4, L, D) so we can split components
        emb = emb.view(B, 4, L, -1)
        # (B,4,L,D) ordering is now explicit
        on = emb[:, 0, :, :]
        nu = emb[:, 1, :, :]
        co = emb[:, 2, :, :]
        to = emb[:, 3, :, :]

        if self.composition == "sum":
            out = on + nu + co + to
        else:  # concat_linear
            cat = torch.cat([on, nu, co], dim=-1)
            out = self.composer(cat) + to  # tone as residual

        # pf_mask: (B, L) True where any component != 0
        pf_mask = (pf4 != 0).any(dim=1)
        return out, pf_mask

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        phone_token: torch.Tensor,
        phone_token_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = "",
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device

        # Concatenate prompt and main text tokens (same as Qwen2LM)
        prompt_len = 0 if prompt_text is None else prompt_text.shape[1]
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len

        # Get token embeddings for the full sequence (prompt + text)
        text_emb = self.qwen2lm.llm.model.model.embed_tokens(text)

        # If phoneme inpainting buffers are provided, compose phoneme embeddings and
        # replace the corresponding embeddings in the *text* portion (i.e., excluding prompt)
        if phone_token is not None:
            # Move to device and basic validation
            phone_token = phone_token.to(device)
            if phone_token.dim() != 2:
                raise ValueError(
                    "phone_token must be shape (B, 4 * L_text_part) or padded flattened (B, 4 * P)"
                )
            B, PT_len = phone_token.shape

            total_len = text_emb.shape[1]
            text_part_len = total_len - prompt_len
            if text_part_len <= 0:
                raise ValueError(
                    "No text tokens available to align phone_token (text part length <= 0)"
                )

            # Compose phoneme embeddings for the text part. The compose_phoneme method
            # supports both padded flattened inputs (with phone_token_len) and
            # full-length flattened inputs (4*L_text_part). If phone_token_len is
            # provided, pass it through; otherwise require PT_len == 4 * text_part_len.
            if phone_token_len is not None:
                phone_token_len = phone_token_len.to(device)
                composed_part, pf_mask_part = self.compose_phoneme(
                    phone_token, phone_token_len
                )
            else:
                if PT_len != 4 * text_part_len:
                    raise ValueError(
                        f"phone_token length {PT_len} does not match 4 * text_part_len ({4 * text_part_len}); provide phone_token_len for variable-length phoneme arrays"
                    )
                composed_part, pf_mask_part = self.compose_phoneme(phone_token, None)

            # Ensure dtype and device match text_emb
            composed_part = composed_part.to(device=device, dtype=text_emb.dtype)
            pf_mask_part = pf_mask_part.to(device=device)

            # If composed_part is shorter/longer than text_part_len, trim or pad as needed
            Lcomp = composed_part.shape[1]
            if Lcomp > text_part_len:
                composed_part = composed_part[:, :text_part_len, :]
                pf_mask_part = pf_mask_part[:, :text_part_len]
            elif Lcomp < text_part_len:
                pad_len = text_part_len - Lcomp
                pad_emb = torch.zeros(
                    (B, pad_len, composed_part.shape[2]),
                    device=device,
                    dtype=composed_part.dtype,
                )
                composed_part = torch.cat([composed_part, pad_emb], dim=1)
                pad_mask = torch.zeros(
                    (B, pad_len), device=device, dtype=pf_mask_part.dtype
                )
                pf_mask_part = torch.cat([pf_mask_part, pad_mask], dim=1)

            # Build full-length composed buffer with zeros for the prompt part
            if prompt_len > 0:
                pad_prompt = torch.zeros(
                    (B, prompt_len, composed_part.shape[2]),
                    device=device,
                    dtype=composed_part.dtype,
                )
                composed_total = torch.cat([pad_prompt, composed_part], dim=1)
                pad_pf = torch.zeros(
                    (B, prompt_len), device=device, dtype=pf_mask_part.dtype
                )
                pf_mask_total = torch.cat([pad_pf, pf_mask_part], dim=1)
            else:
                composed_total = composed_part
                pf_mask_total = pf_mask_part

            # Zero-out text embeddings at phoneme-inpainted positions and add composed phoneme embeddings
            text_emb = text_emb * (~pf_mask_total).unsqueeze(-1).float()
            text_emb = text_emb + composed_total * pf_mask_total.unsqueeze(-1).float()

        # 3. concat llm_input
        sos_emb = self.qwen2lm.llm_embedding.weight[self.qwen2lm.sos].reshape(1, 1, -1)
        task_id_emb = self.qwen2lm.llm_embedding.weight[self.qwen2lm.task_id].reshape(
            1, 1, -1
        )
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.qwen2lm.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.qwen2lm.llm_input_size, dtype=text_emb.dtype
            ).to(device)
        lm_input = torch.concat(
            [sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1
        )

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        for token in self.qwen2lm.inference_wrapper(
            lm_input, sampling, min_len, max_len, uuid
        ):
            yield token

    def forward(self, batch: dict, device: torch.device):
        """A drop-in forward that mirrors `Qwen2LM.forward` but supports per-token phoneme replacement.

        Optional batch keys (see class docstring):
        - `phone_token` (LongTensor `(B, 4*L)`) containing interleaved per-token components `[onset_0, nucleus_0, coda_0, tone_0, onset_1, ...]`.

        Behavior: if `phone_token` is present, it will be split into four `(B, L)` tensors
        and used to compute composed phoneme embeddings which replace the corresponding
        positions in `text_token_emb` prior to building the LM input / target.
        """
        # mirror the Qwen2LM forward behavior up to text_token_emb
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        phone_token = batch["phone_token"].to(device)
        tone_ids = phone_token[:, 3::4]  # (B, L)

        # 1. encode text_token
        text_token_emb = self.qwen2lm.llm.model.model.embed_tokens(text_token)

        # if a merged phone_token is provided, split it into components and integrate phoneme embeddings
        if phone_token.dim() != 2:
            raise ValueError("`phone_token` must be shape (B, 4 * L)")

        B, PT_len = phone_token.shape
        _, text_len = text_token.shape
        if PT_len != 4 * text_len:
            raise ValueError(
                "`phone_token` second dimension must equal 4 * text_token length"
            )
        phone_token_len = batch.get("phone_token_len", None)
        if phone_token_len is not None:
            phone_token_len = phone_token_len.to(device)
            if phone_token_len.dim() != 1 or phone_token_len.numel() != B:
                raise ValueError(
                    "`phone_token_len` must be shape (B,) matching batch size"
                )
            if (phone_token_len > text_len).any():
                raise ValueError(
                    "`phone_token_len` values must be <= text_token length (padded)"
                )

        # Compose phoneme embeddings (returns composed (B, L, D) and a mask pf_mask (B, L))
        composed, pf_mask = self.compose_phoneme(phone_token, phone_token_len)

        # Tone labels for auxiliary tone prediction loss
        tone_mask = (tone_ids != 0) & pf_mask  # (B, L)
        tone_labels = tone_ids - self.tone_offset
        # set ignore_index for CE
        tone_labels_ce = tone_labels.clone()
        tone_labels_ce[~tone_mask] = -100
        # This forces tone signal to live inside phoneme embedding, not context.
        tone_logits = self.tone_classifier(
            composed + 0.0 * text_token_emb.detach()
        )  # (B, L, num_tones)
        # compute CE loss
        tone_logits_flat = tone_logits.view(-1, self.num_tones)
        tone_labels_flat = tone_labels_ce.view(-1)

        tone_loss = F.cross_entropy(
            tone_logits_flat, tone_labels_flat, ignore_index=-100
        )

        # Zero out text embeddings at phoneme-inpainted positions
        text_token_emb = text_token_emb * (~pf_mask).unsqueeze(-1).float()

        # Ensure composed is zero where no components exist
        composed = composed * pf_mask.unsqueeze(-1).float()

        # Final per-token embedding is text + composed phoneme (text zeros at phoneme slots)
        text_token_emb = text_token_emb + composed

        # 3. sos and task_id
        sos_emb = self.qwen2lm.llm_embedding.weight[self.qwen2lm.sos].reshape(1, 1, -1)
        task_id_emb = self.qwen2lm.llm_embedding.weight[self.qwen2lm.task_id].reshape(
            1, 1, -1
        )

        # 2. encode speech_token
        speech_token_emb = self.qwen2lm.speech_embedding(speech_token)

        # 3. prepare llm_input/target (delegate to original helper)
        lm_target, lm_input, lm_input_len = self.qwen2lm.prepare_lm_input_target(
            sos_emb,
            text_token,
            text_token_emb,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_emb,
            speech_token_len,
        )
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.qwen2lm.llm(lm_input, lm_input_len.to(device))
        logits = self.qwen2lm.llm_decoder(lm_output)
        lm_loss = self.qwen2lm.criterion_ce(logits, lm_target.to(device))
        acc = F.one_hot(
            torch.argmax(logits, dim=-1), num_classes=self.qwen2lm.speech_token_size + 3
        ).float()  # placeholder for accuracy
        # Reuse the Qwen2LM's accuracy function if available
        try:
            from cosyvoice.llm.llm import th_accuracy  # type: ignore

            acc = th_accuracy(
                logits.view(-1, self.qwen2lm.speech_token_size + 3),
                lm_target,
                ignore_label=-100,
            )
        except Exception:
            # fall back to coarse accuracy reporting: fraction of non-IGNORE_ID correct tokens
            acc = torch.tensor(0.0, device=device)

        loss = lm_loss + self.tone_weight * tone_loss

        return {
            "loss": loss,
            "acc": acc,
            "lm_loss": lm_loss.detach(),
            "tone_loss": tone_loss.detach(),
        }

    # small utility: initialize component embeddings from the average text embedding
    def init_component_from_text_embed(self):
        """Initialize phoneme component embeddings from the average token embedding of the LLM's tokenizer.

        This helps keep composed embeddings on-manifold as a simple heuristic.
        """
        with torch.no_grad():
            text_emb_weights = self.qwen2lm.llm.model.model.embed_tokens.weight
            avg = text_emb_weights.mean(dim=0)
            # init unified phone embedding: random noise then add avg to non-pad entries
            nn.init.normal_(self.phone_emb.weight, mean=0.0, std=0.02)
            self.phone_emb.weight.data[0].zero_()  # keep padding idx zero
            # Ensure we don't exceed phone_emb's size when adding avg
            if self.phone_emb.weight.data.shape[0] > 1:
                self.phone_emb.weight.data[1:] += avg.unsqueeze(0)
