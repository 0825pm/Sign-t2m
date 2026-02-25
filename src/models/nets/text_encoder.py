import os
import pickle
import torch
import torch.nn as nn
from typing import List, Dict

import clip
from transformers import RobertaTokenizer, RobertaModel, logging
logging.set_verbosity_error()

from ..utils.utils import lengths_to_mask


class Roberta(torch.nn.Module):
    def __init__(self, freeze_lm=True, max_text_len=32):
        super(Roberta, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.lm = RobertaModel.from_pretrained("roberta-base")
        self.max_text_len = max_text_len
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

    def forward(self, text, device, **kwargs):
        text_len = max(min(min([len(x) for x in text]), self.max_text_len), 2) + 1
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=text_len,
        ).to(device)
        out = self.lm(**encoded_input)
        return {"text_emb": out["pooler_output"], "hidden": out["last_hidden_state"],
                "mask": encoded_input["attention_mask"].bool()}


class CLIP(torch.nn.Module):
    def __init__(self, freeze_lm=True):
        super(CLIP, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=torch.device('cpu'), jit=False)
        self.clip_model.visual = None
        if freeze_lm:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    @property
    def dtype(self):
        return self.clip_model.text_projection.dtype

    def forward(self, text, device, **kwargs):
        is_null = [t == "" for t in text]
        text_for_clip = [t if t != "" else "a" for t in text]

        tokens = clip.tokenize(text_for_clip, truncate=True).to(device)
        x = self.clip_model.token_embedding(tokens).type(self.dtype)
        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.dtype)

        text_embed = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.clip_model.text_projection
        mask = lengths_to_mask(tokens.argmax(dim=-1), device)
        if mask.shape[1] < x.shape[1]:
            x = x[:, :mask.size(1)]

        for i, null in enumerate(is_null):
            if null:
                text_embed[i] = 0.0
                x[i] = 0.0
                mask[i] = False
                mask[i, 0] = True

        return {"text_emb": text_embed, "hidden": x, "mask": mask}


class MBartTextEncoder(torch.nn.Module):
    """mBart encoder-only text encoder for multilingual sign language.

    CLIP과 동일한 출력 인터페이스:
        {'text_emb': [B, output_dim], 'hidden': [B, L, output_dim], 'mask': [B, L]}

    SOKE 방식 완전 재현:
        1. map_ids.pkl 로드: {token_id → pruned_emb_id} 매핑
        2. correct_lang_token: 마지막 실제 토큰 → 언어 토큰 교체
        3. map_ids(token_to_emb): token_id → emb_id 변환 후 encoder 전달

    핵심: SOKE mBart는 vocab pruning된 모델.
          tokenizer vocab size (250K+) != embedding size (pruned).
          반드시 map_ids로 리매핑 후 encoder에 전달해야 함.
    """
    SRC2LANG = {
        'how2sign': 'en_XX',
        'csl':      'zh_CN',
        'phoenix':  'de_DE',
        '':         'en_XX',  # CFG uncond fallback
    }

    def __init__(self, model_path: str, freeze_lm: bool = True, output_dim: int = 512):
        super().__init__()
        from transformers import MBartTokenizer, MBartModel

        self.tokenizer = MBartTokenizer.from_pretrained(model_path, legacy=True)
        self.encoder = MBartModel.from_pretrained(model_path).encoder
        if freeze_lm:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # 1024 -> output_dim projection (denoiser text_dim=512 유지)
        self.proj = nn.Linear(1024, output_dim)

        # SOKE map_ids.pkl 로드
        # vocab pruning 후 {token_id: emb_id} 딕셔너리.
        # tokenizer token_id -> pruned embedding index 변환에 사용.
        map_ids_path = os.path.join(model_path, 'map_ids.pkl')
        with open(map_ids_path, 'rb') as f:
            tok_id_to_emb_id: dict = pickle.load(f)  # {int: int}

        # unk token의 emb_id — 매핑 실패 시 fallback
        unk_tok_id = self.tokenizer.convert_tokens_to_ids('<unk>')
        self._unk_emb_id = tok_id_to_emb_id.get(unk_tok_id, 3)

        # dict -> LUT(LongTensor): GPU forward에서 빠른 인덱싱
        max_tok_id = max(tok_id_to_emb_id.keys())
        lut = torch.full((max_tok_id + 1,), self._unk_emb_id, dtype=torch.long)
        for tok_id, emb_id in tok_id_to_emb_id.items():
            lut[tok_id] = emb_id
        self.register_buffer('tok_to_emb_lut', lut)  # [max_token_id+1]

    def _map_token_to_emb(self, input_ids: torch.Tensor) -> torch.Tensor:
        """SOKE map_ids(direction='token_to_emb') 재현.
        token_id가 LUT 범위를 벗어나면 unk_emb_id로 대체.
        """
        ids = input_ids.clamp(0, self.tok_to_emb_lut.shape[0] - 1)
        return self.tok_to_emb_lut[ids]

    def forward(self, text: List[str], device, srcs: List[str] = None, **kwargs):
        is_null = [t == "" for t in text]
        text_in = [t if t else "a" for t in text]  # 빈 문자열 -> dummy

        enc = self.tokenizer(
            text_in,
            padding='longest',
            max_length=256,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt',
        )
        input_ids = enc.input_ids.to(device)            # [B, L], token_id space
        attention_mask = enc.attention_mask.to(device)  # [B, L]

        # Step 1: correct_lang_token
        # 마지막 실제 토큰을 데이터셋별 언어 토큰으로 교체 (SOKE 방식)
        token_lengths = attention_mask.sum(-1)  # [B]
        srcs_ = srcs if srcs is not None else [''] * len(text)
        for i, (src, length) in enumerate(zip(srcs_, token_lengths)):
            lang = self.SRC2LANG.get(src, 'en_XX')
            lang_id = self.tokenizer.convert_tokens_to_ids(lang)
            if lang_id is not None and lang_id != self.tokenizer.unk_token_id:
                input_ids[i, length - 1] = lang_id

        # Step 2: map_ids(token_to_emb)
        # token_id -> pruned embedding index (SOKE 핵심)
        input_ids = self._map_token_to_emb(input_ids)  # [B, L], emb_id space

        # Step 3: encoder forward
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # [B, L, 1024]

        # mean pooling -> projection
        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(1) / mask_f.sum(1)  # [B, 1024]
        text_emb = self.proj(pooled)                        # [B, output_dim]
        hidden_proj = self.proj(hidden)                     # [B, L, output_dim]

        # CFG uncond zero-out
        for i, null in enumerate(is_null):
            if null:
                text_emb[i] = 0.0
                hidden_proj[i] = 0.0
                attention_mask[i] = False
                attention_mask[i, 0] = True  # attention 최소 1 토큰 유지

        return {
            'text_emb': text_emb,
            'hidden':   hidden_proj,
            'mask':     attention_mask.bool(),
        }