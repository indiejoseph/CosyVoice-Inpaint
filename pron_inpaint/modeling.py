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
  - `phoneme_token` with shape `(B, 4 * L)` where `L` is the `text_token` length. The layout is grouped by component:
    `[onset_0, ..., onset_{L-1}, nucleus_0, ..., nucleus_{L-1}, coda_0, ..., coda_{L-1}, tone_0, ..., tone_{L-1}]`.
    The implementation will split this into four `(B, L)` component tensors `(onset, nucleus, coda, tone)`. An index of `0` means "no phoneme".
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
        onset_vocab: size of onset vocab (include 0 for "no onset")
        nucleus_vocab: likewise for nucleus
        coda_vocab: likewise for coda
        tone_vocab: likewise for tone
        d_model: embedding dimension (should match qwen2lm.llm_input_size)
        composition: one of {'concat_linear', 'sum'} (default 'concat_linear')
        gate_blend: if True, use a learned gate to blend phoneme and text embeddings
    """

    def __init__(
        self,
        qwen2lm,
        onset_vocab: int,
        nucleus_vocab: int,
        coda_vocab: int,
        tone_vocab: int,
        d_model: int,
        composition: str = "concat_linear",
        gate_blend: bool = False,
    ):
        super().__init__()
        self.qwen2lm = qwen2lm
        self.d_model = d_model
        assert composition in ("concat_linear", "sum")
        self.composition = composition
        self.gate_blend = gate_blend

        # component embedding tables. index 0 is reserved as "no phoneme" / padding
        self.onset_emb = nn.Embedding(onset_vocab, d_model, padding_idx=0)
        self.nucleus_emb = nn.Embedding(nucleus_vocab, d_model, padding_idx=0)
        self.coda_emb = nn.Embedding(coda_vocab, d_model, padding_idx=0)
        self.tone_emb = nn.Embedding(tone_vocab, d_model, padding_idx=0)

        if composition == "concat_linear":
            self.composer = nn.Sequential(
                nn.Linear(4 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.composer = None

        if gate_blend:
            # gate takes [text_emb, phoneme_emb] -> scalar in (0,1)
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )

    def compose_phoneme(
        self,
        onset: torch.LongTensor,
        nucleus: torch.LongTensor,
        coda: torch.LongTensor,
        tone: torch.LongTensor,
    ) -> torch.Tensor:
        """Compose phoneme embeddings per token.

        Inputs are expected to be LongTensors of shape (B, L). Returns a FloatTensor
        (B, L, d_model).
        """
        # lookup component embeddings -> (B, L, D)
        on = self.onset_emb(onset)
        nu = self.nucleus_emb(nucleus)
        co = self.coda_emb(coda)
        to = self.tone_emb(tone)

        if self.composition == "sum":
            out = on + nu + co + to
        else:  # concat_linear
            cat = torch.cat([on, nu, co, to], dim=-1)
            out = self.composer(cat)
        return out

    def replace_text_embeddings(
        self,
        text_emb: torch.Tensor,
        onset: torch.LongTensor,
        nucleus: torch.LongTensor,
        coda: torch.LongTensor,
        tone: torch.LongTensor,
    ) -> torch.Tensor:
        """Return `text_emb` with phoneme-composed embeddings replacing selected tokens.

        Replacement mask: positions where any component index != 0.
        """
        composed = self.compose_phoneme(onset, nucleus, coda, tone)
        # mask positions where any component is non-zero
        mask = (onset != 0) | (nucleus != 0) | (coda != 0) | (tone != 0)
        mask = mask.unsqueeze(-1)  # (B, L, 1)

        if self.gate_blend:
            gate_in = torch.cat([text_emb, composed], dim=-1)
            gate = torch.sigmoid(self.gate(gate_in))  # (B, L, D)
            blended = gate * composed + (1.0 - gate) * text_emb
            out = torch.where(mask, blended, text_emb)
        else:
            out = torch.where(mask, composed, text_emb)
        return out

    def forward(self, batch: dict, device: torch.device):
        """A drop-in forward that mirrors `Qwen2LM.forward` but supports per-token phoneme replacement.

        Optional batch keys (see class docstring):
        - `phoneme_token` (LongTensor `(B, 4*L)`) containing grouped components `[onset..., nucleus..., coda..., tone...]`.

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

        # if a merged phoneme_token is provided, split it into components and replace per-token embeddings
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
            L = text_len
            # layout: [onset_0..onset_{L-1}, nucleus_0..nucleus_{L-1}, coda_0..coda_{L-1}, tone_0..tone_{L-1}]
            onset = phoneme_token[:, 0:L]
            nucleus = phoneme_token[:, L : 2 * L]
            coda = phoneme_token[:, 2 * L : 3 * L]
            tone = phoneme_token[:, 3 * L : 4 * L]
            # ensure shapes are (B, L)
            if onset.shape != text_token.shape:
                raise ValueError(
                    "`phoneme_token` should split into (B, L) component tensors matching `text_token.shape`"
                )
            text_token_emb = self.replace_text_embeddings(
                text_token_emb, onset, nucleus, coda, tone
            )

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
        """Initialize phoneme component embeddings to the average token embedding of the LLM's tokenizer.

        This helps keep composed embeddings on-manifold as a simple heuristic.
        """
        with torch.no_grad():
            text_emb_weights = self.qwen2lm.llm.model.model.embed_tokens.weight
            avg = text_emb_weights.mean(dim=0)
            for emb in (self.onset_emb, self.nucleus_emb, self.coda_emb, self.tone_emb):
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)
                emb.weight.data[0].zero_()  # keep padding idx zero
                emb.weight.data[1:] += avg.unsqueeze(0)
