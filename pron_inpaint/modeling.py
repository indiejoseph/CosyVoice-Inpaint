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
  - `phoneme_token` with shape `(B, 4 * L)` where `L` is the `text_token` length. The layout is interleaved per token:
    `[onset_0, nucleus_0, coda_0, tone_0, onset_1, nucleus_1, coda_1, tone_1, ...]`.
    The implementation will rearrange this into four `(B, L)` component tensors `(onset, nucleus, coda, tone)`. An index of `0` means "no phoneme".
  - If this key is absent, behavior falls back to standard Qwen2LM forward.
- Multi-syllable insertion (multiple phoneme embeddings per text token)
  is *not* supported by this initial implementation. It is documented and may
  be extended later.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen2LMInpaint(nn.Module):
    """Wraps a `Qwen2LM` instance and provides phoneme inpainting utilities.

    Args:
        qwen2lm: an instantiated Qwen2LM (or CosyVoice3LM) from third_party/CosyVoice
        vocab_size: total phoneme vocabulary size (includes pad=0 and covers all components)
        d_model: embedding dimension (should match qwen2lm.llm_input_size)
        composition: one of {'concat_linear', 'sum'} (default 'concat_linear')
    """

    def __init__(
        self,
        qwen2lm,
        vocab_size: int,
        d_model: int,
        composition: str = "concat_linear",
    ):
        """Note: `vocab_size` must be the total phoneme vocabulary size including pad idx 0.
        For example: vocab_size = (n_onset + n_nucleus + n_coda + n_tone + 1)
        """
        super().__init__()
        self.qwen2lm = qwen2lm
        self.d_model = d_model
        assert composition in ("concat_linear", "sum")
        self.composition = composition

        # unified phoneme embedding table: index 0 is reserved as "no phoneme" / padding
        total_vocab = vocab_size
        self.phone_emb = nn.Embedding(total_vocab, d_model, padding_idx=0)
        self.composer = torch.nn.Identity()
        if composition == "concat_linear":
            self.composer = nn.Sequential(
                nn.Linear(4 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

    def compose_phoneme(
        self, phoneme_flat: torch.LongTensor, phoneme_token_len: torch.LongTensor = None
    ) -> tuple:
        """Compose phoneme embeddings from a flattened phoneme id tensor.

        Args:
            phoneme_flat: LongTensor of shape (B, 4*L) where the layout is interleaved per token:
                [onset_0, nucleus_0, coda_0, tone_0, onset_1, nucleus_1, ...]
            phoneme_token_len: optional (B,) unpadded per-sample text token lengths

        Returns:
            tuple((B, L, d_model) composed phoneme embeddings, (B, L) boolean mask indicating which token positions contain any phoneme component)

        Behavior:
            If `phoneme_token_len` is provided, the method will reconstruct a
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
        if phoneme_token_len is not None:
            if phoneme_token_len.dim() != 1 or phoneme_token_len.numel() != B:
                raise ValueError(
                    "phoneme_token_len must be shape (B,) matching batch size"
                )
            Lmax = L
            pf4 = torch.zeros(
                (B, 4, Lmax), dtype=phoneme_flat.dtype, device=phoneme_flat.device
            )
            for i in range(B):
                Li = int(phoneme_token_len[i].item())
                if Li < 0 or Li > Lmax:
                    raise ValueError(
                        "phoneme_token_len values must be between 0 and Lmax"
                    )
                if 4 * Li > PT_len:
                    raise ValueError(
                        "phoneme_flat too short for declared phoneme_token_len"
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
            cat = torch.cat([on, nu, co, to], dim=-1)
            out = self.composer(cat)

        # pf_mask: (B, L) True where any component != 0
        pf_mask = (pf4 != 0).any(dim=1)
        return out, pf_mask

    def forward(self, batch: dict, device: torch.device):
        """A drop-in forward that mirrors `Qwen2LM.forward` but supports per-token phoneme replacement.

        Optional batch keys (see class docstring):
        - `phoneme_token` (LongTensor `(B, 4*L)`) containing interleaved per-token components `[onset_0, nucleus_0, coda_0, tone_0, onset_1, ...]`.

        Behavior: if `phoneme_token` is present, it will be split into four `(B, L)` tensors
        and used to compute composed phoneme embeddings which replace the corresponding
        positions in `text_token_emb` prior to building the LM input / target.
        """
        # mirror the Qwen2LM forward behavior up to text_token_emb
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)

        # 1. encode text_token
        text_token_emb = self.qwen2lm.llm.model.model.embed_tokens(text_token)

        # if a merged phoneme_token is provided, split it into components and integrate phoneme embeddings
        if "phoneme_token" in batch and batch["phoneme_token"] is not None:
            phoneme_token = batch["phoneme_token"].to(device)
            if phoneme_token.dim() != 2:
                raise ValueError("`phoneme_token` must be shape (B, 4 * L)")
            B, PT_len = phoneme_token.shape
            _, text_len = text_token.shape
            if PT_len != 4 * text_len:
                raise ValueError(
                    "`phoneme_token` second dimension must equal 4 * text_token length"
                )
            phoneme_token_len = batch.get("phoneme_token_len", None)
            if phoneme_token_len is not None:
                phoneme_token_len = phoneme_token_len.to(device)
                if phoneme_token_len.dim() != 1 or phoneme_token_len.numel() != B:
                    raise ValueError(
                        "`phoneme_token_len` must be shape (B,) matching batch size"
                    )
                if (phoneme_token_len > text_len).any():
                    raise ValueError(
                        "`phoneme_token_len` values must be <= text_token length (padded)"
                    )

            # Compose phoneme embeddings (returns composed (B, L, D) and a mask pf_mask (B, L))
            composed, pf_mask = self.compose_phoneme(phoneme_token, phoneme_token_len)

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
        loss = self.qwen2lm.criterion_ce(logits, lm_target.to(device))
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

        return {"loss": loss, "acc": acc}

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
