"""
MambaVae - Single-token Mamba VAE for Sign Language Motion
Ported from SignGPT3/motGPT/archs/mamba_vae.py

Latent: z [1, B, 256] — 120-dim 전체를 하나의 토큰으로 압축
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
from typing import List, Optional, Tuple

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[MambaVae] Warning: mamba_ssm not installed. Using Conv1D fallback.")


def lengths_to_mask(lengths, device, max_len=None):
    if max_len is None:
        max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < torch.tensor(lengths, device=device).unsqueeze(1)
    return mask


# =============================================================================
# Building Blocks
# =============================================================================
class LocalModule(nn.Module):
    def __init__(self, dim, num_groups=16, kernel_size=3):
        super().__init__()
        num_groups = min(num_groups, dim)
        while dim % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()

    def forward(self, x, mask=None):
        residual = x
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        x = x + residual
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(dim)
        if MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            if bidirectional:
                self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
                self.combine = nn.Linear(dim * 2, dim)
            self._is_mamba = True
        else:
            inner_dim = dim * expand
            self.conv_block = nn.Sequential(
                nn.Conv1d(dim, inner_dim, 1), nn.SiLU(),
                nn.Conv1d(inner_dim, inner_dim, d_conv, padding=d_conv // 2, groups=inner_dim), nn.SiLU(),
                nn.Conv1d(inner_dim, dim, 1),
            )
            self._is_mamba = False

    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        if self._is_mamba:
            if self.bidirectional:
                fwd = self.mamba_fwd(x)
                bwd = self.mamba_bwd(x.flip(1)).flip(1)
                x = self.combine(torch.cat([fwd, bwd], dim=-1))
            else:
                x = self.mamba_fwd(x)
        else:
            x = x.transpose(1, 2)
            x = self.conv_block(x)
            x = x.transpose(1, 2)
        x = x + residual
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


# =============================================================================
# Encoder: Motion [B,T,120] → z [1,B,256]
# =============================================================================
class MambaEncoder(nn.Module):
    def __init__(self, input_dim=120, latent_dim=256, num_layers=4,
                 num_groups=16, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim),
            nn.SiLU(), nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 512, latent_dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'local': LocalModule(latent_dim, num_groups),
                'mamba': MambaBlock(latent_dim, d_state, d_conv, expand, bidirectional=True),
            }))
        self.final_norm = nn.LayerNorm(latent_dim)
        self.pool_query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(latent_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.to_mu = nn.Linear(latent_dim, latent_dim)
        self.to_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        device = x.device
        mask = lengths_to_mask(lengths, device, max_len=T)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        for layer in self.layers:
            x = layer['local'](x, mask)
            x = layer['mamba'](x, mask)
        x = self.final_norm(x)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, x, x, key_padding_mask=~mask)
        pooled = pooled.squeeze(1)
        mu = self.to_mu(pooled).unsqueeze(0)       # [1, B, D]
        logvar = self.to_logvar(pooled).unsqueeze(0)  # [1, B, D]
        return mu, logvar


# =============================================================================
# Decoder: z [1,B,256] → Motion [B,T,120]
# =============================================================================
class MambaDecoder(nn.Module):
    def __init__(self, output_dim=120, latent_dim=256, num_layers=4,
                 num_groups=16, d_state=16, d_conv=4, expand=2, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, latent_dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'mamba': MambaBlock(latent_dim, d_state, d_conv, expand, bidirectional=True),
                'local': LocalModule(latent_dim, num_groups),
            }))
        self.final_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, output_dim)

    def forward(self, z, lengths):
        if z.dim() == 3:
            z = z.squeeze(0)  # [1,B,D] → [B,D]
        B = z.shape[0]
        T = max(lengths)
        device = z.device
        mask = lengths_to_mask(lengths, device, max_len=T)
        z = self.latent_proj(z)
        x = z.unsqueeze(1).expand(-1, T, -1)  # broadcast
        x = x + self.pos_embed[:, :T, :]
        for layer in self.layers:
            x = layer['mamba'](x, mask)
            x = layer['local'](x, mask)
        x = self.final_norm(x)
        x = self.output_proj(x)
        x = x * mask.unsqueeze(-1).float()
        return x


# =============================================================================
# MambaVae (Main)
# =============================================================================
class MambaVae(nn.Module):
    """
    Single-token Mamba VAE.
    z: [1, B, 256]

    Interface:
        z, dist = vae.encode(features, lengths)   # z: [1, B, 256]
        recon = vae.decode(z, lengths)             # recon: [B, T, 120]
    """
    def __init__(self, nfeats=120, latent_dim=256, num_layers=4, num_groups=16,
                 d_state=16, d_conv=4, expand=2, dropout=0.1,
                 ablation=None, ff_size=1024, num_heads=4, arch=None,
                 normalize_before=False, activation='gelu', position_embedding='learned',
                 datatype='h2s', **kwargs):
        super().__init__()
        if isinstance(latent_dim, (list, tuple)):
            self.latent_size = latent_dim[0]
            self.latent_dim = latent_dim[-1]
        else:
            self.latent_size = 1
            self.latent_dim = latent_dim

        self.nfeats = nfeats
        self.mean_std_inv = 0.8457
        self.mean_std_inv_2 = self.mean_std_inv ** 2
        self.mean_mean = -0.1379

        self.encoder = MambaEncoder(
            input_dim=nfeats, latent_dim=self.latent_dim, num_layers=num_layers,
            num_groups=num_groups, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout,
        )
        self.decoder_net = MambaDecoder(
            output_dim=nfeats, latent_dim=self.latent_dim, num_layers=num_layers,
            num_groups=num_groups, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout,
        )
        total = sum(p.numel() for p in self.parameters())
        print(f"[MambaVae] nfeats={nfeats}, latent=[{self.latent_size},{self.latent_dim}], "
              f"layers={num_layers}, params={total/1e6:.2f}M, Mamba={MAMBA_AVAILABLE}")

    def encode(self, features, lengths=None):
        if lengths is None:
            lengths = [features.shape[1]] * features.shape[0]
        mu, logvar = self.encoder(features, lengths)
        std = (logvar * 0.5).exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        return z, dist

    def decode(self, z, lengths):
        return self.decoder_net(z, lengths)

    def forward(self, features, lengths=None):
        z, dist = self.encode(features, lengths)
        recon = self.decode(z, lengths)
        return recon, z, dist
