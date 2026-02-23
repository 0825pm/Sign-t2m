"""
SignDenoiser — Mamba-based sequence-level denoiser for motion-space diffusion
Ported from Light-T2M's LightT2M (nets/light_final.py)

Input:  noisy motion [B, T, motion_dim], padding_mask, timestep, text_embed
Output: predicted motion/noise [B, T, motion_dim]
"""

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from functools import partial

from einops import rearrange, repeat, reduce
from torch import Tensor

from src.models.utils.embedding import timestep_embedding, PositionEmbedding

from mamba_ssm import Mamba


# =========================================================================
# Mamba Block (from Light-T2M)
# =========================================================================
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class BaseMambaBlock(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm,
        fused_add_norm=False, residual_in_fp32=False, pre_norm=True,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        self.pre_norm = pre_norm
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm))

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None,
        inference_params=None, text_len=None, **mixer_kwargs,
    ):
        if not self.pre_norm:
            return self.post_norm_forward(hidden_states, residual, inference_params, **mixer_kwargs)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states, self.norm.weight, self.norm.bias,
                residual=residual, prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm),
                text_len=text_len,
            )
        hidden_states = self.mixer(
            hidden_states, inference_params=inference_params,
            text_len=text_len, **mixer_kwargs,
        )
        return hidden_states, residual

    def post_norm_forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None,
        inference_params=None, **mixer_kwargs,
    ):
        new_hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        if not self.fused_add_norm:
            new_hidden_states = self.norm(hidden_states + new_hidden_states)
        else:
            new_hidden_states = layer_norm_fn(
                new_hidden_states, self.norm.weight, self.norm.bias,
                residual=hidden_states, prenorm=False, residual_in_fp32=False,
                eps=self.norm.eps, is_rms_norm=isinstance(self.norm, RMSNorm),
            )
        return new_hidden_states, None


def create_mamba_block(
    d_model, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False,
    residual_in_fp32=False, fused_add_norm=False,
    layer_idx=None, device=None, dtype=None, pre_norm=True,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon, **factory_kwargs,
    )
    block = BaseMambaBlock(
        d_model, mixer_cls, norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        pre_norm=pre_norm,
    )
    block.layer_idx = layer_idx
    return block


# =========================================================================
# Sub-modules
# =========================================================================

# --- Part-Aware I/O (preserves body/hand structure in hidden space) ---

class PartAwareInputProj(nn.Module):
    """Project each body part separately → hidden dims proportional to part size.
    
    120D motion: body(30D) / lhand(45D) / rhand(45D)
    → hidden:    body(64D) / lhand(96D) / rhand(96D) = 256D total
    
    This preserves part boundaries in the hidden space so downstream
    PartAwareLocalModule can process each part with dedicated parameters.
    """
    def __init__(self, motion_splits, hidden_dims):
        super().__init__()
        self.motion_splits = [int(s) for s in motion_splits]
        self.projs = nn.ModuleList([
            nn.Linear(ms, hd) for ms, hd in zip(motion_splits, hidden_dims)
        ])

    def forward(self, x):
        parts = torch.split(x, self.motion_splits, dim=-1)
        return torch.cat([proj(p) for p, proj in zip(parts, self.projs)], dim=-1)


class PartAwareOutputProj(nn.Module):
    """Project back from part-structured hidden space to motion dims."""
    def __init__(self, hidden_dims, motion_splits):
        super().__init__()
        self.hidden_dims = [int(d) for d in hidden_dims]
        self.projs = nn.ModuleList([
            nn.Linear(hd, ms) for hd, ms in zip(hidden_dims, motion_splits)
        ])

    def forward(self, x):
        parts = torch.split(x, self.hidden_dims, dim=-1)
        return torch.cat([proj(p) for p, proj in zip(parts, self.projs)], dim=-1)


# --- Part-Aware Local Module ---

class PartAwareLocalModule(nn.Module):
    """Part-aware local temporal modeling for sign language.
    
    Inspired by SALAD's Skeleton Attention + SOKE's Decoupled processing.
    
    1. Split hidden features by part boundaries (body/lhand/rhand)
    2. Per-part temporal Conv1d (intra-part: each hand learns its own dynamics)
    3. Cross-part gating (inter-part: hands informed by body context)
    
    vs original LocalModule which treats all dims uniformly with a single Conv1d.
    """
    def __init__(self, model_dim, part_dims, num_groups=8, mask_padding=True):
        super().__init__()
        self.mask_padding = mask_padding
        self.part_dims = [int(d) for d in part_dims]
        self.num_parts = len(part_dims)

        # Per-part temporal convolution (dedicated parameters per body part)
        self.part_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(pd, pd, 1, 1, 0),
                nn.Conv1d(pd, pd, 3, 1, 1, groups=pd),
                nn.GroupNorm(num_groups=min(num_groups, pd // 4), num_channels=pd),
                nn.SiLU(),
            ) for pd in part_dims
        ])

        # Cross-part gating: each part sees global context to modulate itself
        # e.g., hand motion should be aware of body pose for anatomical consistency
        total_dim = sum(part_dims)
        self.cross_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, pd),
                nn.Sigmoid(),
            ) for pd in part_dims
        ])

        self.norm = nn.LayerNorm(total_dim)

    def forward(self, x, x_mask, y, y_mask, z=None):
        if self.mask_padding:
            x[~x_mask] = 0
            y[~y_mask] = 0

        # Split into parts along feature dim
        parts = torch.split(x, self.part_dims, dim=-1)

        # Per-part temporal convolution
        conv_parts = []
        for part, conv in zip(parts, self.part_convs):
            conv_parts.append(conv(part.permute(0, 2, 1)).permute(0, 2, 1))

        # Cross-part gating: concatenate all → gate each part
        concat_all = torch.cat(conv_parts, dim=-1)
        gated_parts = []
        for cp, gate in zip(conv_parts, self.cross_gates):
            gated_parts.append(cp * gate(concat_all))

        merged = torch.cat(gated_parts, dim=-1)
        x = self.norm(x + merged)
        return x, y


# --- Original LocalModule (kept for backward compatibility) ---

class LocalModule(nn.Module):
    def __init__(self, model_dim, num_groups=16, mask_padding=True):
        super().__init__()
        self.mask_padding = mask_padding
        self.conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1, 1, 0),
            nn.Conv1d(model_dim, model_dim, 3, 1, 1, groups=model_dim),
            nn.GroupNorm(num_groups=num_groups, num_channels=model_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, x_mask, y, y_mask, z=None):
        if self.mask_padding:
            x[~x_mask] = x[~x_mask] * torch.zeros_like(x[~x_mask])
            y[~y_mask] = y[~y_mask] * torch.zeros_like(y[~y_mask])
        x = self.norm(x + self.conv(x.permute(0, 2, 1)).permute(0, 2, 1))
        return x, y


class MixedModule(nn.Module):
    def __init__(self, model_dim, build_mamba_block_fn, patch_size=8, mask_padding=True):
        super().__init__()
        self.patch_size = patch_size
        self.mask_padding = mask_padding

        self.local_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, patch_size, patch_size, 0, groups=model_dim),
        )
        self.global_mamba = build_mamba_block_fn(model_dim)
        self.final_fc = nn.Linear(model_dim * 2, model_dim)
        self.norm = nn.LayerNorm(model_dim)


        self.f_func = nn.Linear(model_dim * 2, model_dim)
        self.fuse_fn = nn.Linear(model_dim * 2, model_dim)

    def inject_text(self, x, y):
        y_repeat = y.repeat(1, x.shape[1], 1)
        y_hat = self.f_func(torch.cat([x, y_repeat], dim=-1))
        _y_hat = y_repeat * torch.sigmoid_(y_hat)
        x_hat = self.fuse_fn(torch.cat([x, _y_hat], dim=-1))
        return x_hat

    def forward(self, x, x_mask, y, y_mask):
        if self.mask_padding:
            x[~x_mask] = x[~x_mask] * torch.zeros_like(x[~x_mask])
            y[~y_mask] = y[~y_mask] * torch.zeros_like(y[~y_mask])

        B, L, D = x.shape
        x1 = x[:, 1:]  # skip time token
        padding_size = x1.shape[1] % self.patch_size
        if padding_size != 0:
            x1 = torch.cat(
                [x1, torch.zeros(B, self.patch_size - padding_size, D, device=x1.device)],
                dim=1,
            )

        nx1 = self.local_conv(x1.permute(0, 2, 1)).permute(0, 2, 1)


        x2 = self.inject_text(nx1, y)

        nx2, _ = self.global_mamba(torch.cat([x2.flip([1]), x2], dim=1))
        x2 = nx2[:, x2.shape[1]:]
        x2 = repeat(x2, "B L D -> B (L S) D", S=self.patch_size)

        nx = self.final_fc(torch.cat([x1, x2], dim=-1))
        if padding_size != 0:
            nx = nx[:, : -(self.patch_size - padding_size)]

        out = torch.cat([x[:, :1], nx], dim=1)
        out = self.norm(out)
        return out, y


class StageBlock(nn.Module):
    def __init__(
        self, in_dim, dim, build_mamba_block_fn, mask_padding,
        num_groups=16, patch_size=8, part_aware=False, part_dims=None,
    ):
        super().__init__()
        if part_aware and part_dims is not None:
            self.local_module1 = PartAwareLocalModule(dim, part_dims, num_groups=8, mask_padding=mask_padding)
            self.local_module2 = PartAwareLocalModule(dim, part_dims, num_groups=8, mask_padding=mask_padding)
        else:
            self.local_module1 = LocalModule(dim, num_groups, mask_padding)
            self.local_module2 = LocalModule(dim, num_groups, mask_padding)
        self.mixed_module = MixedModule(dim, build_mamba_block_fn, patch_size, mask_padding)

        self.input_proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()
        self.y_proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()

    def forward(self, x, x_mask, y, y_mask):
        x = self.input_proj(x)
        y_ = self.y_proj(y)
        x, _ = self.local_module1(x, x_mask, y_, y_mask)
        x, _ = self.mixed_module(x, x_mask, y_, y_mask)
        x, _ = self.local_module2(x, x_mask, y_, y_mask)
        return x, y


# =========================================================================
# Main Denoiser
# =========================================================================
class SignDenoiser(nn.Module):
    """
    Mamba-based denoiser for motion-space diffusion.
    Same architecture as Light-T2M's LightT2M.

    Input:  motion [B, T, motion_dim], padding_mask [B, T], timestep [B], text_embed
    Output: predicted [B, T, motion_dim]
    """

    def __init__(
        self,
        motion_dim=120,
        max_motion_len=301,
        text_dim=512,
        pos_emb="cos",
        dropout=-1,
        stage_dim="256*4",
        num_groups=16,
        patch_size=8,
        ssm_cfg=None,
        rms_norm=False,
        fused_add_norm=True,
        # Part-aware options (sign language)
        part_aware=False,
        motion_splits=None,   # e.g. [30, 45, 45] for body/lhand/rhand
    ):
        super().__init__()
        self.part_aware = part_aware
        self.motion_dim = motion_dim

        # Parse stage_dim
        if "*" in str(stage_dim):
            base_dim = int(str(stage_dim).split("*")[0])
            stage_dims = [base_dim] * int(str(stage_dim).split("*")[1])
        else:
            stage_dims = [int(x) for x in str(stage_dim).split("-")]
            base_dim = stage_dims[0]

        # Part-aware: compute hidden dims proportional to motion part sizes
        if part_aware and motion_splits is not None:
            self.motion_splits = [int(s) for s in motion_splits]
            total_motion = sum(motion_splits)
            # Allocate hidden dims proportionally, ensure they sum to base_dim
            raw = [int(base_dim * ms / total_motion) for ms in motion_splits]
            raw[-1] = base_dim - sum(raw[:-1])  # fix rounding
            self.part_hidden_dims = raw  # e.g. [64, 96, 96] for 256
            print(f"[SignDenoiser] Part-aware: motion_splits={motion_splits} → hidden_dims={self.part_hidden_dims}")
        else:
            self.motion_splits = None
            self.part_hidden_dims = None

        # Position embedding
        if pos_emb == "cos":
            self.pos_emb = PositionEmbedding(max_motion_len, base_dim, dropout=0.1)
        elif pos_emb == "learn":
            self.pos_emb = PositionEmbedding(max_motion_len, base_dim, dropout=0.1, grad=True)
        else:
            raise ValueError(f"{pos_emb} not supported!")

        # Projections — part-aware or flat
        if part_aware and self.part_hidden_dims is not None:
            self.m_input_proj = PartAwareInputProj(motion_splits, self.part_hidden_dims)
        else:
            self.m_input_proj = nn.Linear(motion_dim, base_dim)

        self.t_input_proj = nn.Linear(text_dim, base_dim)
        self.time_emb = nn.Linear(base_dim, base_dim)

        # Mamba stages
        create_fn = partial(
            create_mamba_block, ssm_cfg=ssm_cfg, norm_epsilon=1e-5,
            rms_norm=rms_norm, residual_in_fp32=False,
            fused_add_norm=fused_add_norm, pre_norm=False,
        )

        modules = []
        cur_in_dim = base_dim
        for cur_dim in stage_dims:
            # Compute part_dims for this stage's hidden dimension
            stage_part_dims = None
            if part_aware and self.motion_splits is not None:
                total_motion = sum(self.motion_splits)
                raw = [int(cur_dim * ms / total_motion) for ms in self.motion_splits]
                raw[-1] = cur_dim - sum(raw[:-1])
                stage_part_dims = raw
            modules.append(
                StageBlock(cur_in_dim, cur_dim, create_fn,
                           mask_padding=True, num_groups=num_groups, patch_size=patch_size,
                           part_aware=part_aware, part_dims=stage_part_dims)
            )
            cur_in_dim = cur_dim
        self.layers = nn.ModuleList(modules)

        # Output projection — part-aware or flat
        if part_aware and self.part_hidden_dims is not None:
            final_part_dims = None
            total_motion = sum(self.motion_splits)
            raw = [int(cur_in_dim * ms / total_motion) for ms in self.motion_splits]
            raw[-1] = cur_in_dim - sum(raw[:-1])
            final_part_dims = raw
            if dropout > 0:
                self.m_output_proj = nn.Sequential(
                    nn.Dropout(dropout),
                    PartAwareOutputProj(final_part_dims, self.motion_splits),
                )
            else:
                self.m_output_proj = PartAwareOutputProj(final_part_dims, self.motion_splits)
        else:
            if dropout > 0:
                self.m_output_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(cur_in_dim, motion_dim))
            else:
                self.m_output_proj = nn.Linear(cur_in_dim, motion_dim)

    def forward(self, motion, motion_mask, timestep, text):
        """
        Args:
            motion: [B, T, motion_dim] noisy motion
            motion_mask: [B, T] bool padding mask
            timestep: [B] diffusion timestep
            text: dict with 'text_emb' [B, text_dim]
        """
        motion = self.m_input_proj(motion)
        time_emb = self.time_emb(timestep_embedding(timestep, motion.shape[-1])).unsqueeze(1)
        time_mask = torch.ones(time_emb.shape[0], 1, dtype=torch.bool, device=time_emb.device)

        text_feat = self.t_input_proj(text["text_emb"]).unsqueeze(1)
        text_mask = torch.ones(text_feat.shape[0], 1, dtype=torch.bool, device=text_feat.device)


        x = torch.cat([time_emb, motion], dim=1)
        x_mask = torch.cat([time_mask, motion_mask], dim=1)
        x = self.pos_emb(x)

        for layer in self.layers:
            x, text_feat = layer(x, x_mask, text_feat, text_mask)

        out = x[:, 1:]  # remove time token
        out = self.m_output_proj(out)
        return out