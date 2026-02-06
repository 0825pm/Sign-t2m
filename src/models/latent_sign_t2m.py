"""
LatentSignT2M — Stage 2 training module (latent diffusion)
Single-token VAE: z [1, B, 256]
Multi-GPU (DDP) compatible

v2: Motion space loss + Latent normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from lightning.pytorch import LightningModule
from hydra.utils import instantiate
from diffusers import DDPMScheduler, UniPCMultistepScheduler

from src.models.sign_vae import MambaVae
from src.models.nets.latent_sign_denoiser import LatentSignDenoiser


class LatentSignT2M(LightningModule):
    """
    Latent Diffusion for Sign Language T2M.

    Components:
        - VAE (frozen): Motion [B,T,120] ↔ z [1,B,256]
        - Text Encoder (frozen): CLIP
        - Denoiser (trainable): z_noisy + text → z_pred
    
    v2 Changes:
        - Latent normalization (z → N(0,1))
        - Motion space loss (optional)
    """
    def __init__(
        self,
        # sub-configs (Hydra already instantiates these)
        text_encoder=None,
        denoiser=None,
        noise_scheduler=None,
        sample_scheduler=None,
        optimizer=None,
        lr_scheduler=None,
        ema=None,
        # VAE
        vae_checkpoint: str = "",
        vae_config: dict = None,
        # generation
        text_replace_prob: float = 0.1,
        guidance_scale: float = 4.0,
        step_num: int = 10,
        # NEW: loss weights
        lambda_motion: float = 0.5,  # motion space loss weight
        use_motion_loss: bool = True,
        # misc
        dataset_name: str = "sign",
        evaluator=None,
        save_every_n_epochs: float = 1e9,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['text_encoder', 'denoiser'])

        # --- Text Encoder (frozen) ---
        self.text_encoder = text_encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # --- Noise / Sample Schedulers ---
        self.noise_scheduler = noise_scheduler
        if sample_scheduler is False:
            self.sample_scheduler = noise_scheduler
            self.sample_scheduler.set_timesteps(1000)
        else:
            self.sample_scheduler = sample_scheduler
            self.sample_scheduler.set_timesteps(step_num)

        # --- Denoiser (trainable) ---
        if isinstance(denoiser, nn.Module):
            self.denoiser = denoiser
        else:
            from omegaconf import OmegaConf
            if hasattr(denoiser, '_metadata'):
                d_cfg = OmegaConf.to_container(denoiser, resolve=True)
            else:
                d_cfg = dict(denoiser) if denoiser else {}
            self.denoiser = LatentSignDenoiser(**d_cfg)

        # --- VAE (frozen) ---
        self.vae = self._load_vae(vae_checkpoint, vae_config)

        # --- Latent normalization (lazy init) ---
        # Will be computed on first batch or loaded from checkpoint
        self.register_buffer('z_mean', torch.zeros(256))
        self.register_buffer('z_std', torch.ones(256))
        self._z_stats_initialized = False

        # --- CFG: null text embedding (lazy init) ---
        self.text_replace_prob = text_replace_prob
        self._null_emb = None

        # --- Loss weights ---
        self.lambda_motion = lambda_motion
        self.use_motion_loss = use_motion_loss

        # EMA placeholder
        self.ema_denoiser = None

    def _load_vae(self, ckpt_path, vae_config):
        """Load pre-trained VAE and freeze."""
        if vae_config is None:
            vae_config = {}

        from omegaconf import OmegaConf
        if hasattr(vae_config, '_metadata'):
            vae_config = OmegaConf.to_container(vae_config, resolve=True)
        else:
            vae_config = dict(vae_config)

        vae = MambaVae(**vae_config)
        if ckpt_path:
            print(f"[LatentSignT2M] Loading VAE from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            cleaned = {}
            for k, v in state.items():
                k2 = k.replace('vae.', '') if k.startswith('vae.') else k
                cleaned[k2] = v
            vae.load_state_dict(cleaned, strict=False)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        return vae

    @property
    def null_emb(self):
        if self._null_emb is None:
            with torch.no_grad():
                out = self.text_encoder([""], self.device)
                self._null_emb = out["text_emb"].detach()
        return self._null_emb

    # =========================================================================
    # Latent Normalization
    # =========================================================================
    def _compute_z_stats(self, dataloader, max_batches=100):
        """Compute latent mean/std from training data."""
        print("[LatentSignT2M] Computing latent statistics...")
        all_z = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                motion = batch['motion'].to(self.device)
                lengths = batch['motion_len'].tolist()
                z, _ = self.vae.encode(motion, lengths)  # [1, B, 256]
                all_z.append(z.squeeze(0))  # [B, 256]
        
        all_z = torch.cat(all_z, dim=0)  # [N, 256]
        self.z_mean = all_z.mean(dim=0)
        self.z_std = all_z.std(dim=0).clamp(min=1e-6)
        self._z_stats_initialized = True
        
        print(f"  z_mean: {self.z_mean.mean():.4f} (range: {self.z_mean.min():.4f} ~ {self.z_mean.max():.4f})")
        print(f"  z_std:  {self.z_std.mean():.4f} (range: {self.z_std.min():.4f} ~ {self.z_std.max():.4f})")

    def normalize_z(self, z):
        """Normalize latent to ~N(0,1)"""
        return (z - self.z_mean.to(z.device)) / self.z_std.to(z.device)

    def denormalize_z(self, z_norm):
        """Denormalize latent back to original scale"""
        return z_norm * self.z_std.to(z_norm.device) + self.z_mean.to(z_norm.device)

    def on_fit_start(self):
        """Compute z stats at the start of training."""
        if not self._z_stats_initialized:
            train_loader = self.trainer.datamodule.train_dataloader()
            self._compute_z_stats(train_loader, max_batches=100)

    # =========================================================================
    # Training
    # =========================================================================
    def training_step(self, batch, batch_idx):
        motion = batch['motion']       # [B, T, 120]
        text = batch['text']           # List[str]
        lengths = batch['motion_len']  # [B] tensor
        lengths_list = lengths.tolist()

        # Encode motion → latent
        with torch.no_grad():
            z_raw, _ = self.vae.encode(motion, lengths_list)  # [1, B, 256]
        z_raw = z_raw.squeeze(0).unsqueeze(1)  # [B, 1, 256]
        
        # Normalize latent
        z = self.normalize_z(z_raw)

        # Text encoding
        with torch.no_grad():
            text_out = self.text_encoder(text, self.device)
            text_emb = text_out["text_emb"].float()

        # CFG: randomly replace text with null
        if self.text_replace_prob > 0 and self.training:
            mask = torch.rand(text_emb.shape[0], device=self.device) < self.text_replace_prob
            if mask.any():
                text_emb[mask] = self.null_emb.to(text_emb.dtype)

        # Diffusion forward
        noise = torch.randn_like(z)
        B = z.shape[0]
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device)
        z_noisy = self.noise_scheduler.add_noise(z, noise, t)

        # Predict
        z_pred = self.denoiser(z_noisy, t, text_emb)  # [B, 1, 256]

        # Loss
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'sample':
            # 1. Latent space loss (normalized)
            loss_latent = F.mse_loss(z_pred, z)
            
            # 2. Motion space loss
            if self.use_motion_loss and self.lambda_motion > 0:
                # Denormalize and decode
                z_pred_raw = self.denormalize_z(z_pred)
                z_pred_dec = z_pred_raw.squeeze(1).unsqueeze(0)  # [1, B, 256]
                motion_pred = self.vae.decode(z_pred_dec, lengths_list)  # [B, T, 120]
                
                # Motion loss with length mask
                loss_motion = 0
                for i, L in enumerate(lengths_list):
                    loss_motion += F.mse_loss(motion_pred[i, :L], motion[i, :L])
                loss_motion = loss_motion / B
                
                loss = loss_latent + self.lambda_motion * loss_motion
                self.log('train/loss_motion', loss_motion, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            else:
                loss = loss_latent
            
            self.log('train/loss_latent', loss_latent, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        else:
            loss = F.mse_loss(z_pred, noise)

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        motion = batch['motion']
        text = batch['text']
        lengths = batch['motion_len']
        lengths_list = lengths.tolist()

        with torch.no_grad():
            z_raw, _ = self.vae.encode(motion, lengths_list)
        z_raw = z_raw.squeeze(0).unsqueeze(1)
        z = self.normalize_z(z_raw)

        with torch.no_grad():
            text_out = self.text_encoder(text, self.device)
            text_emb = text_out["text_emb"].float()

        noise = torch.randn_like(z)
        B = z.shape[0]
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device)
        z_noisy = self.noise_scheduler.add_noise(z, noise, t)
        z_pred = self.denoiser(z_noisy, t, text_emb)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'sample':
            loss_latent = F.mse_loss(z_pred, z)
            
            if self.use_motion_loss and self.lambda_motion > 0:
                z_pred_raw = self.denormalize_z(z_pred)
                z_pred_dec = z_pred_raw.squeeze(1).unsqueeze(0)
                motion_pred = self.vae.decode(z_pred_dec, lengths_list)
                
                loss_motion = 0
                for i, L in enumerate(lengths_list):
                    loss_motion += F.mse_loss(motion_pred[i, :L], motion[i, :L])
                loss_motion = loss_motion / B
                
                loss = loss_latent + self.lambda_motion * loss_motion
                self.log('val/loss_motion', loss_motion, prog_bar=True, on_epoch=True, sync_dist=True)
            else:
                loss = loss_latent
            
            self.log('val/loss_latent', loss_latent, prog_bar=True, on_epoch=True, sync_dist=True)
        else:
            loss = F.mse_loss(z_pred, noise)

        self.log('val/loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

    # =========================================================================
    # Inference
    # =========================================================================
    @torch.no_grad()
    def generate(self, text: List[str], lengths: List[int]):
        B = len(text)
        device = self.device

        # Text encode (with CFG duplication)
        text_out = self.text_encoder(text, device)
        cond_emb = text_out["text_emb"].float()
        uncond_emb = self.null_emb.expand(B, -1).to(cond_emb.dtype)
        text_emb = torch.cat([cond_emb, uncond_emb], dim=0)  # [2B, 512]

        # Init noise (in normalized space)
        z = torch.randn(B, 1, self.denoiser.latent_dim, device=device)
        z = z * self.sample_scheduler.init_noise_sigma

        self.sample_scheduler.set_timesteps(self.hparams.step_num)
        pred_type = self.noise_scheduler.config.prediction_type

        for t in self.sample_scheduler.timesteps.to(device):
            z_input = z.repeat(2, 1, 1)  # [2B, 1, 256]
            output = self.denoiser(z_input, t.expand(2 * B), text_emb)

            if pred_type == 'sample':
                cond_x0, uncond_x0 = output.chunk(2)
                # CFG on x0 directly
                pred_x0 = uncond_x0 + self.hparams.guidance_scale * (cond_x0 - uncond_x0)
                z = self.sample_scheduler.step(pred_x0, t, z).prev_sample.float()
            else:
                cond_eps, uncond_eps = output.chunk(2)
                pred_noise = uncond_eps + self.hparams.guidance_scale * (cond_eps - uncond_eps)
                z = self.sample_scheduler.step(pred_noise, t, z).prev_sample.float()

        # Denormalize before decoding
        z_raw = self.denormalize_z(z)
        z_dec = z_raw.squeeze(1).unsqueeze(0)  # [B,1,256] → [1,B,256]
        motion = self.vae.decode(z_dec, lengths)
        return motion

    # =========================================================================
    # Optimizer
    # =========================================================================
    def configure_optimizers(self):
        opt_cfg = self.hparams.optimizer
        if callable(opt_cfg):
            opt = opt_cfg(params=self.denoiser.parameters())
        else:
            opt = instantiate(opt_cfg, params=self.denoiser.parameters())

        ret = {"optimizer": opt}
        if self.hparams.lr_scheduler is not None:
            sched_cfg = self.hparams.lr_scheduler
            if callable(sched_cfg):
                sched = sched_cfg(optimizer=opt)
            else:
                sched = instantiate(sched_cfg, optimizer=opt)
            ret["lr_scheduler"] = {"scheduler": sched, "interval": "epoch"}
        return ret