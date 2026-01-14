import sys
import threading
import time
import torch
import uuid
import numpy as np
from typing import Generator, TYPE_CHECKING

sys.path.append("third_party/CosyVoice")

if TYPE_CHECKING:
    from cosyvoice.cli.cosyvoice import CosyVoice2Model


def cosyvoice2_tts(
    self,
    text=torch.zeros(1, 0, dtype=torch.int32),
    phone=torch.zeros(1, 0, dtype=torch.int32),
    flow_embedding=torch.zeros(0, 192),
    llm_embedding=torch.zeros(0, 192),
    prompt_text=torch.zeros(1, 0, dtype=torch.int32),
    llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
    flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
    prompt_speech_feat=torch.zeros(1, 0, 80),
    source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
    stream=False,
    speed=1.0,
    **kwargs
):
    # this_uuid is used to track variables related to this inference thread
    this_uuid = str(uuid.uuid1())
    with self.lock:
        self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
        self.hift_cache_dict[this_uuid] = None
    if source_speech_token.shape[1] == 0:
        p = threading.Thread(
            target=self.llm_job,
            args=(
                text,
                phone,
                prompt_text,
                llm_prompt_speech_token,
                llm_embedding,
                this_uuid,
            ),
        )
    else:
        p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
    p.start()
    if stream is True:
        token_offset = 0
        prompt_token_pad = int(
            np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len)
            * self.token_hop_len
            - flow_prompt_speech_token.shape[1]
        )
        while True:
            time.sleep(0.1)
            this_token_hop_len = (
                self.token_hop_len + prompt_token_pad
                if token_offset == 0
                else self.token_hop_len
            )
            if (
                len(self.tts_speech_token_dict[this_uuid]) - token_offset
                >= this_token_hop_len + self.flow.pre_lookahead_len
            ):
                this_tts_speech_token = torch.tensor(
                    self.tts_speech_token_dict[this_uuid][
                        : token_offset
                        + this_token_hop_len
                        + self.flow.pre_lookahead_len
                    ]
                ).unsqueeze(dim=0)
                this_tts_speech = self.token2wav(
                    token=this_tts_speech_token,
                    prompt_token=flow_prompt_speech_token,
                    prompt_feat=prompt_speech_feat,
                    embedding=flow_embedding,
                    token_offset=token_offset,
                    uuid=this_uuid,
                    stream=stream,
                    finalize=False,
                )
                token_offset += this_token_hop_len
                yield {"tts_speech": this_tts_speech.cpu()}
            if (
                self.llm_end_dict[this_uuid] is True
                and len(self.tts_speech_token_dict[this_uuid]) - token_offset
                < this_token_hop_len + self.flow.pre_lookahead_len
            ):
                break
        p.join()
        # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
        this_tts_speech_token = torch.tensor(
            self.tts_speech_token_dict[this_uuid]
        ).unsqueeze(dim=0)
        this_tts_speech = self.token2wav(
            token=this_tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            token_offset=token_offset,
            uuid=this_uuid,
            finalize=True,
        )
        yield {"tts_speech": this_tts_speech.cpu()}
    else:
        # deal with all tokens
        p.join()
        this_tts_speech_token = torch.tensor(
            self.tts_speech_token_dict[this_uuid]
        ).unsqueeze(dim=0)
        this_tts_speech = self.token2wav(
            token=this_tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            token_offset=0,
            uuid=this_uuid,
            finalize=True,
            speed=speed,
        )
        yield {"tts_speech": this_tts_speech.cpu()}
    with self.lock:
        self.tts_speech_token_dict.pop(this_uuid)
        self.llm_end_dict.pop(this_uuid)
        self.hift_cache_dict.pop(this_uuid)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.current_stream().synchronize()


def cosyvoice2_llm_job(
    self, text, phone, prompt_text, llm_prompt_speech_token, llm_embedding, uuid
):
    cur_silent_token_num, max_silent_token_num = 0, 5
    with self.llm_context, torch.cuda.amp.autocast(
        self.fp16 is True and hasattr(self.llm, "vllm") is False
    ):
        if isinstance(text, Generator):
            raise NotImplementedError(
                "Generator input for text is not supported in cosyvoice_llm_job."
            )
        else:
            token_generator = self.llm.inference(
                text=text.to(self.device),
                text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(
                    self.device
                ),
                phone_token=phone.to(self.device),
                phone_token_len=torch.tensor(
                    [phone.shape[1] // 4], dtype=torch.int32
                ).to(self.device),
                prompt_text=prompt_text.to(self.device),
                prompt_text_len=torch.tensor(
                    [prompt_text.shape[1]], dtype=torch.int32
                ).to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                prompt_speech_token_len=torch.tensor(
                    [llm_prompt_speech_token.shape[1]], dtype=torch.int32
                ).to(self.device),
                embedding=llm_embedding.to(self.device),
                uuid=uuid,
            )
        for i in token_generator:
            if i in self.silent_tokens:
                cur_silent_token_num += 1
                if cur_silent_token_num > max_silent_token_num:
                    continue
            else:
                cur_silent_token_num = 0
            self.tts_speech_token_dict[uuid].append(i)
    self.llm_end_dict[uuid] = True


def patch_cosyvoice(cosyvoice: "CosyVoice2Model"):
    from pron_inpaint.frontend_wrapper import InpaintFrontendWrapper
    from pron_inpaint.modeling import Qwen2LMInpaint

    inpaint_frontend = InpaintFrontendWrapper(
        frontend=cosyvoice.frontend, device=cosyvoice.model.device
    )
    phone_vocab_size = inpaint_frontend.phone_tokenizer.vocab_size()
    cosyvoice.frontend = inpaint_frontend
    cosyvoice.model.llm_job = cosyvoice2_llm_job.__get__(cosyvoice.model)
    cosyvoice.model.tts = cosyvoice2_tts.__get__(cosyvoice.model)
    cosyvoice.model.llm = Qwen2LMInpaint(
        cosyvoice.model.llm, phone_vocab_size, composition="concat_linear"
    ).to(cosyvoice.model.device)
