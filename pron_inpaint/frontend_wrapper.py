"""Frontend for Jyutping annotation.

Example: "你好呀[aa3]！"

Output:
Text: "你好呀！"
Phoneme: "你好 aa3 !"
"""

import sys
import json
import re
from functools import partial
import torch
from typing import TYPE_CHECKING, Generator, Optional
import inflect

sys.path.append("../third_party/CosyVoice")

from cosyvoice.utils.frontend_utils import (
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    split_paragraph,
    is_only_punctuation,
    contains_chinese,
)
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
from cosyvoice.utils.file_utils import logging
from pron_inpaint.tokenizer import JyutpingTokenizer

if TYPE_CHECKING:
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from pron_inpaint.tokenizer import JyutpingTokenizer


class InpaintFrontendWrapper:
    """
    Wrapper for frontend, following the interface of CosyVoiceFrontEnd.

    Responsibilities:
    - Normalize input text for TTS while preserving inline Jyutping phoneme annotations
      written as square-bracketed tokens (e.g., "你好[aa3]！").
    - Replace bracketed phonemes with the frontend tokenizer's pad token in the
      text tokens so downstream components can detect phoneme insertion points.
    - Extract interleaved phoneme sequences aligned to padded token positions and
      return `(phone_token, phone_token_len)` suitable for the inpainting model.

    Phoneme layout convention (interleaved per-token):
      [onset_0, nucleus_0, coda_0, tone_0, onset_1, nucleus_1, ...]

    Returned phone tensors:
    - `phone_token`: tensor shaped `(1, text_len*4)` where every text token has 4
      consecutive slots [onset, nucleus, coda, tone]. Slots without phonemes are 0.
    - `phone_token_len`: tensor `(1,)` equal to `phone_token.shape[1]` (text_len*4)

    Note: This wrapper expects `phone_tokenizer.encode([phoneme_str])` to return a
    flat list of interleaved phoneme ids (length = phoneme_count * 4).
    """

    def __init__(
        self,
        frontend: Optional["CosyVoiceFrontEnd"] = None,
        tokenizer_path: Optional[str] = None,
        device=torch.device("cpu"),
    ):
        self.frontend = frontend
        self.frontend_tokenizer = None
        self.phone_tokenizer = JyutpingTokenizer()
        self.text_frontend = "wetext"
        self.device = device
        self.allowed_special = "all"
        self.inflect_parser = inflect.engine()
        self.zh_tn_model = None
        self.en_tn_model = None
        self.frd = None

        if self.frontend is not None:
            self.text_frontend = self.frontend.text_frontend
            self.frontend_tokenizer = self.frontend.tokenizer
            self.device = self.frontend.device
            self.allowed_special = self.frontend.allowed_special
            self.inflect_parser = self.frontend.inflect_parser

            if self.frontend.text_frontend == "wetext":
                from wetext import Normalizer as ZhNormalizer

                self.frontend.zh_tn_model = ZhNormalizer(
                    remove_erhua=False,
                    full_to_half=False,
                    traditional_to_simple=False,
                    remove_interjections=False,
                )
                self.zh_tn_model = self.frontend.zh_tn_model
                self.en_tn_model = self.frontend.en_tn_model
            elif self.frontend.text_frontend == "ttsfrd":
                self.frd = self.frontend.frd

        else:
            if tokenizer_path is not None:
                self.frontend_tokenizer = get_qwen_tokenizer(
                    tokenizer_path, skip_special_tokens=True, version="cosyvoice2"
                )
            else:
                raise ValueError(
                    "Either `frontend` or `text_tokenizer` must be provided."
                )

            try:
                from wetext import Normalizer as ZhNormalizer
                from wetext import Normalizer as EnNormalizer

                self.zh_tn_model = ZhNormalizer(
                    remove_erhua=False,
                    full_to_half=False,
                    traditional_to_simple=False,
                    remove_interjections=False,
                )
                self.en_tn_model = EnNormalizer()
                self.text_frontend = "wetext"
                logging.info("use wetext frontend")
            except Exception as e:
                self.text_frontend = ""
                logging.info("no frontend is avaliable")

    def text_normalize(self, text: str, split=True, text_frontend=True):
        """
        Normalize and optionally split input text for TTS while preserving bracketed phonemes.

        Behavior:
        - Supports both Chinese and English normalization paths.
        - Keeps inline bracketed Jyutping annotations (e.g., "你好[aa3]！") in the
          returned text so they can be extracted later by `_extract_phone_token`.

        Args:
            text: raw input text (may include bracketed phonemes)
            split: whether to return a list of split segments
            text_frontend: whether to run the full frontend normalization (set False
                for SSML or pre-normalized text)

        Returns:
            A list of normalized text segments if `split=True`, otherwise a single string.
        """
        if isinstance(text, Generator):
            logging.info("get tts_text generator, will skip text_normalize!")
            return [text]

        lang = "zh" if contains_chinese(text) else "en"

        # NOTE skip text_frontend when ssml symbol in text
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend is False or text == "":
            return [text] if split is True else text

        def split_by_brackets(input_string):
            # Extract phonemes inside brackets and the surrounding parts outside of them.
            # Example: "你[wa1]好[hui2]" -> inside_brackets=["wa1","hui2"],
            # outside_brackets=["你", "好", ""]
            inside_brackets = re.findall(r"\[([a-z]+[1-6]{1})\]", input_string)
            outside_brackets = re.split(r"\[[a-z]+[1-6]{1}\]", input_string)

            # Filter out empty strings from the outside list (result of consecutive brackets)
            outside_brackets = [part for part in outside_brackets if part]

            return inside_brackets, outside_brackets

        def join_interleaved(outside, inside):
            # Combine alternating outside/inside parts preserving bracketed phonemes.
            # If there are more outside parts than inside (e.g., trailing text), append it.
            result = []

            # Iterate and combine alternating parts
            for o, i in zip(outside, inside):
                result.append(o + "[" + i + "]")

            # Append any remaining part (if outside is longer than inside)
            if len(outside) > len(inside):
                result.append(outside[-1])

            return "".join(result)

        def text_normalize_no_split(text, is_last=False):
            text = text.strip()

            if self.text_frontend == "ttsfrd":
                texts = [
                    i["text"]
                    for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]
                ]
                text = "".join(texts)
            else:
                if lang == "zh":
                    if self.text_frontend == "wetext":
                        text = self.zh_tn_model.normalize(text)
                    text = text.replace("\n", "")
                    text = replace_blank(text)
                    text = replace_corner_mark(text)
                    text = text.replace(".", "。")
                    text = text.replace(" - ", "，")
                    text = remove_bracket(text)
                    text = re.sub(r"[，,、]+$", "。", text)
                    text_is_terminated = text[-1] in {"。", "！", "？"}
                    if not text_is_terminated and not is_last:
                        text = text[:-1]
                else:
                    if self.text_frontend == "wetext":
                        text = self.en_tn_model.normalize(text)
                    text = spell_out_number(text, self.inflect_parser)
            return text

        inside_brackets, outside_brackets = split_by_brackets(text)

        for n in range(len(outside_brackets)):
            e_out = text_normalize_no_split(
                outside_brackets[n], is_last=n == len(outside_brackets) - 1
            )
            outside_brackets[n] = e_out

        text = join_interleaved(outside_brackets, inside_brackets)

        texts = list(
            split_paragraph(
                text,
                partial(
                    self.frontend_tokenizer.encode,
                    allowed_special=self.allowed_special,
                ),
                lang,
                token_max_n=80,
                token_min_n=60,
                merge_len=20,
                comma_split=False,
            )
        )
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def _extract_text_token(self, text):
        """
        Tokenize normalized text for the LLM frontend.

        - Replaces any bracketed phoneme annotations with the tokenizer's pad token so
          downstream token positions can be matched to phoneme slots.

        Returns:
            text_token: tensor of shape (1, text_len) with token ids (pad tokens remain)
            text_token_len: tensor (1,) giving the token length
        """
        if isinstance(text, Generator):
            logging.info(
                "get tts_text generator, will return _extract_text_token_generator!"
            )

            # NOTE add a dummy text_token_len for compatibility
            return self._extract_text_token_generator(text), torch.tensor(
                [0], dtype=torch.int32
            ).to(self.device)
        else:
            # replace phoneme [...] to padding token <pad>
            text = re.sub(r"\[.*?\]", self.frontend_tokenizer.tokenizer.pad_token, text)

            text_token = self.frontend_tokenizer.encode(
                text, allowed_special=self.allowed_special
            )
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(
                self.device
            )
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def _extract_phone_token(self, text, text_token):
        """
        Extract bracketed Jyutping phonemes from `text` and align them to pad positions
        in `text_token`.

        Returns:
            phone_token: tensor shape (1, text_len*4) with interleaved phoneme ids per token
            phone_token_len: tensor (1,) equal to the number of phoneme tokens (P) for the sample.
                - If `phone_token` is already padded to `text_len*4` you can pass `phone_token_len=None` to
                  indicate the buffer is aligned to text tokens.

        Raises:
            AssertionError: if the number of bracketed phonemes does not match the number of pad tokens
        """
        # extract phoneme from [...]
        phonemes = re.findall(r"\[([a-z]+[1-6]{1})\]", text)
        phoneme_count = len(phonemes)
        phoneme_str = " ".join(phonemes) if len(phonemes) > 0 else ""

        # count how many pad tokens in text_token
        pad_token_id = self.frontend_tokenizer.tokenizer.pad_token_id
        pad_count = (text_token[0] == pad_token_id).sum().item()

        assert phoneme_count == pad_count, (
            f"phoneme count {phoneme_count} != pad count {pad_count}, "
            f"please check your input text: {text}"
        )

        if pad_count == 0 or phoneme_str == "":
            # return a dummy phone token when no phoneme
            phone_token = torch.zeros(
                (1, text_token.shape[1] * 4), dtype=torch.int32
            ).to(self.device)
            phone_token_len = torch.tensor(
                [phone_token.shape[1]], dtype=torch.int32
            ).to(self.device)
            return phone_token, phone_token_len

        # `tmp_phone_token` is expected to be shape [1, phoneme_count * 4]
        tmp_phone_token = self.phone_tokenizer.encode(
            [phoneme_str]
        )  # [1, phoneme_count*4]

        # create a buffer aligned to the full text length: each text token has 4 slots
        phone_token = torch.zeros((1, text_token.shape[1] * 4), dtype=torch.int32).to(
            self.device
        )

        # Walk through text tokens and write phoneme chunks into pad token slots
        text_i = 0
        phone_i = 0
        while phone_i < phoneme_count and text_i < text_token.shape[1]:
            # When a pad token is found, place the next 4 phoneme ids into the token's 4 slots
            if text_token[0, text_i] == pad_token_id:
                phone_token[0, text_i * 4 : (text_i + 1) * 4] = torch.tensor(
                    tmp_phone_token[0][phone_i * 4 : (phone_i + 1) * 4],
                    dtype=torch.int32,
                ).to(self.device)
                phone_i += 1
            text_i += 1

        # `phone_token_len` is the number of phoneme tokens (P) present in `phonemes`.
        # This follows the model API where `phoneme_token_len[i] = P_i` allows reconstructing
        # the first 4*P entries when inputs were flattened at original lengths.
        phone_token_len = torch.tensor([phone_token.shape[1]], dtype=torch.int32).to(
            self.device
        )
        return phone_token, phone_token_len

    def frontend_sft(self, tts_text, spk_id):
        if self.frontend is None:
            raise ValueError("frontend is not provided!")

        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        tts_phone_token, tts_phone_token_len = self._extract_phone_token(
            tts_text, tts_text_token
        )
        embedding = self.frontend.spk2info[spk_id]["embedding"]
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "phone_token": tts_phone_token,
            "phone_token_len": tts_phone_token_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_zero_shot(
        self, tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        if self.frontend is None:
            raise ValueError(
                "frontend_zero_shot requires `frontend` to be provided in the wrapper."
            )
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        tts_phone_token, tts_phone_token_len = self._extract_phone_token(
            tts_text, tts_text_token
        )
        if zero_shot_spk_id == "":
            prompt_text_token, prompt_text_token_len = self._extract_text_token(
                prompt_text
            )
            speech_feat, speech_feat_len = self.frontend._extract_speech_feat(
                prompt_wav
            )
            speech_token, speech_token_len = self.frontend._extract_speech_token(
                prompt_wav
            )
            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = (
                    speech_feat[:, : 2 * token_len],
                    2 * token_len,
                )
                speech_token, speech_token_len[:] = (
                    speech_token[:, :token_len],
                    token_len,
                )
            embedding = self.frontend._extract_spk_embedding(prompt_wav)
            model_input = {
                "prompt_text": prompt_text_token,
                "prompt_text_len": prompt_text_token_len,
                "llm_prompt_speech_token": speech_token,
                "llm_prompt_speech_token_len": speech_token_len,
                "flow_prompt_speech_token": speech_token,
                "flow_prompt_speech_token_len": speech_token_len,
                "prompt_speech_feat": speech_feat,
                "prompt_speech_feat_len": speech_feat_len,
                "llm_embedding": embedding,
                "flow_embedding": embedding,
            }
        else:
            model_input = {**self.frontend.spk2info[zero_shot_spk_id]}
        model_input["text"] = tts_text_token
        model_input["text_len"] = tts_text_token_len
        model_input["phone"] = tts_phone_token
        model_input["phone_len"] = tts_phone_token_len

        return model_input

    def frontend_cross_lingual(
        self, tts_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        return self.frontend.frontend_cross_lingual(
            tts_text, prompt_wav, resample_rate, zero_shot_spk_id
        )

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        return self.frontend_instruct(tts_text, spk_id, instruct_text)

    def frontend_instruct2(
        self, tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        return self.frontend_zero_shot(
            tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
        )

    def frontend_vc(self, source_speech_16k, prompt_wav, resample_rate):
        return self.frontend.frontend_vc(source_speech_16k, prompt_wav, resample_rate)
