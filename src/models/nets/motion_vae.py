"""
MLD-style Motion VAE with Skip-Connection Transformers.
Adapted from MotionGPT3 (OpenMotionLab) / MLD (ChenFengYe).

Architecture:
  Encoder: motion [B,T,D] → skel_embedding → SkipTransformerEncoder → mu,logvar → z [1,B,256]
  Decoder: z [1,B,256] → SkipTransformerDecoder (cross-attn) → final_layer → motion [B,T,D]
"""

import copy
import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution


# ============================================================
#  Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (sequence-first: [T, B, D])."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: [T, B, D]"""
        x = x + self.pe[: x.shape[0]]
        return self.dropout(x)


# ============================================================
#  Transformer Layers (with optional pos embedding add)
# ============================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu"):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu"):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=tgt if query_pos is None else tgt + query_pos,
            key=memory if pos is None else memory + pos,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# ============================================================
#  Skip-Connection Transformer (U-Net style)
# ============================================================
class SkipTransformerEncoder(nn.Module):
    """
    U-Net style Transformer encoder with skip connections.
    num_layers must be odd: e.g. 9 → 4 input + 1 middle + 4 output.
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1, f"num_layers must be odd, got {num_layers}"

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = copy.deepcopy(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        x = src
        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x


class SkipTransformerDecoder(nn.Module):
    """
    U-Net style Transformer decoder with skip connections.
    num_layers must be odd.
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = decoder_layer.d_model
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1, f"num_layers must be odd, got {num_layers}"

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = copy.deepcopy(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(
            nn.Linear(2 * self.d_model, self.d_model), num_block
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        x = tgt
        xs = []
        for module in self.input_blocks:
            x = module(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask,
                       pos=pos, query_pos=query_pos)
            xs.append(x)

        x = self.middle_block(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              pos=pos, query_pos=query_pos)

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask,
                       pos=pos, query_pos=query_pos)

        if self.norm is not None:
            x = self.norm(x)
        return x


# ============================================================
#  MLD VAE
# ============================================================
class MldVae(nn.Module):
    """
    Motion Latent Diffusion VAE.
    Encodes variable-length motion [B, T, nfeats] → z [1, B, latent_dim].
    Decodes z + lengths → reconstructed motion [B, T, nfeats].

    Args:
        nfeats:     motion feature dimension (528 for our sign language)
        latent_dim: [latent_size, latent_dim] e.g. [1, 256]
        ff_size:    feedforward hidden dimension
        num_layers: must be odd (skip connection requirement)
        num_heads:  attention heads
        dropout:    dropout rate
        activation: "gelu" or "relu"
    """

    def __init__(
        self,
        nfeats: int = 528,
        latent_dim: list = [1, 256],
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()

        self.latent_size = latent_dim[0]   # number of latent tokens (1)
        self.latent_dim = latent_dim[-1]   # latent feature dim (256)
        self.nfeats = nfeats

        # --- Embedding layers ---
        self.skel_embedding = nn.Linear(nfeats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, nfeats)

        # --- Distribution tokens (learnable) ---
        # 2 tokens: one for mu, one for logvar
        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.latent_dim)
        )

        # --- Positional encoding ---
        self.query_pos_encoder = PositionalEncoding(self.latent_dim, dropout)
        self.query_pos_decoder = PositionalEncoding(self.latent_dim, dropout)

        # --- Encoder (SkipTransformer) ---
        encoder_layer = TransformerEncoderLayer(
            self.latent_dim, num_heads, ff_size, dropout, activation
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)

        # --- Decoder (SkipTransformer with cross-attention) ---
        decoder_layer = TransformerDecoderLayer(
            self.latent_dim, num_heads, ff_size, dropout, activation
        )
        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm)

    def freeze(self):
        """Freeze all parameters (for Stage 2 latent diffusion)."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(
        self,
        features: Tensor,
        lengths: Optional[List[int]] = None,
    ) -> tuple:
        """
        Encode motion features to latent distribution.

        Args:
            features: [B, T, nfeats]
            lengths:  list of valid lengths per batch

        Returns:
            latent: [1, B, latent_dim]  (reparameterized sample)
            dist:   torch.distributions.Normal
        """
        if lengths is None:
            lengths = [features.shape[1]] * features.shape[0]

        device = features.device
        bs, nframes, nfeats = features.shape

        # Lengths mask: [B, T]
        mask = lengths_to_mask(lengths, device, max_len=nframes)

        # Project to latent dim: [B, T, nfeats] → [B, T, latent_dim]
        x = self.skel_embedding(features)

        # [B, T, D] → [T, B, D]  (PyTorch Transformer convention)
        x = x.permute(1, 0, 2)

        # Prepend distribution tokens: [2, B, D]
        dist_tokens = self.global_motion_token[:, None, :].expand(-1, bs, -1)

        # Augmented sequence: [2+T, B, D]
        xseq = torch.cat([dist_tokens, x], dim=0)

        # Augmented mask: [B, 2+T]
        token_mask = torch.ones((bs, dist_tokens.shape[0]), dtype=bool, device=device)
        aug_mask = torch.cat([token_mask, mask], dim=1)

        # Positional encoding + SkipTransformerEncoder
        xseq = self.query_pos_encoder(xseq)
        encoded = self.encoder(xseq, src_key_padding_mask=~aug_mask)

        # Extract distribution tokens: [latent_size, B, D] for mu, same for logvar
        mu = encoded[0:self.latent_size]        # [1, B, 256]
        logvar = encoded[self.latent_size:self.latent_size * 2]  # [1, B, 256]

        # Reparameterize (clamp logvar to prevent fp16 overflow)
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std, validate_args=False)
        latent = dist.rsample()  # [1, B, 256]

        return latent, dist

    def decode(self, z: Tensor, lengths: List[int]) -> Tensor:
        """
        Decode latent to motion features.

        Args:
            z:       [1, B, latent_dim]  (or [latent_size, B, latent_dim])
            lengths: list of valid lengths per batch

        Returns:
            feats: [B, T, nfeats]
        """
        device = z.device
        mask = lengths_to_mask(lengths, device)  # [B, max_T]
        bs, nframes = mask.shape

        # Zero time queries: [T, B, D]
        queries = torch.zeros(nframes, bs, self.latent_dim, device=device)

        # Positional encoding for queries
        queries = self.query_pos_decoder(queries)

        # SkipTransformerDecoder: queries attend to z via cross-attention
        output = self.decoder(
            tgt=queries,
            memory=z,
            tgt_key_padding_mask=~mask,
        )

        # If output has extra dim from decoder, squeeze
        if output.dim() == 4:
            output = output.squeeze(0)

        # Project back to motion space: [T, B, D] → [T, B, nfeats]
        output = self.final_layer(output)

        # Zero out padded positions
        output[~mask.T] = 0

        # [T, B, nfeats] → [B, T, nfeats]
        feats = output.permute(1, 0, 2)
        return feats

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        """
        Full forward pass: encode → reparameterize → decode.
        Returns: (reconstructed_motion, latent, dist)
        """
        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist


# ============================================================
#  Utilities
# ============================================================
def lengths_to_mask(lengths, device, max_len=None):
    """Generate boolean mask from lengths. [B, max_len]"""
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths, device=device)
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")