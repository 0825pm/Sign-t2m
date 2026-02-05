"""
LatentSignDenoiser — Denoiser for single-token latent diffusion
Input: z_noisy [B, 256] + timestep + text_emb → z_pred [B, 256]

Sequence: [time_token, text_token, z_token] → self-attn blocks → extract z
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DenoiserBlock(nn.Module):
    """Self-attention + FFN block with residual"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class LatentSignDenoiser(nn.Module):
    """
    Denoiser for single-token latent space.

    Architecture:
        z_noisy [B, 256] → proj → z_token [B, 1, dim]
        timestep [B]      → emb  → time_token [B, 1, dim]
        text_emb [B, 512] → proj → text_token [B, 1, dim]
                                         ↓
        concat → [B, 3, dim] → DenoiserBlock × N → extract z → [B, 256]
    """
    def __init__(
        self,
        latent_dim: int = 256,
        text_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        # ignored kwargs for config compat
        ssm_cfg: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projections
        self.z_input_proj = nn.Linear(latent_dim, hidden_dim)
        self.z_output_proj = nn.Linear(hidden_dim, latent_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Learnable token type embeddings
        self.token_type_emb = nn.Parameter(torch.randn(1, 3, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([DenoiserBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)

        total = sum(p.numel() for p in self.parameters())
        print(f"[LatentSignDenoiser] dim={hidden_dim}, layers={num_layers}, params={total/1e6:.2f}M")

    def forward(self, z_noisy, timestep, text_emb):
        """
        Args:
            z_noisy: [B, 1, latent_dim] or [B, latent_dim]
            timestep: [B]
            text_emb: [B, text_dim]  (CLIP pooled output)
        Returns:
            z_pred: [B, 1, latent_dim]
        """
        if z_noisy.dim() == 2:
            z_noisy = z_noisy.unsqueeze(1)  # [B, 1, D]

        B = z_noisy.shape[0]

        # Build 3-token sequence
        time_token = self.t_proj(timestep_embedding(timestep, self.hidden_dim)).unsqueeze(1)  # [B,1,dim]
        text_token = self.text_proj(text_emb).unsqueeze(1)     # [B,1,dim]
        z_token = self.z_input_proj(z_noisy)                    # [B,1,dim]

        x = torch.cat([time_token, text_token, z_token], dim=1)  # [B,3,dim]
        x = x + self.token_type_emb

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        z_pred = self.z_output_proj(x[:, 2:3, :])  # extract z token → [B,1,latent_dim]
        return z_pred
