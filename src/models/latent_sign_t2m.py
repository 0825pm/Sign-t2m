"""
LatentSignT2M — Stage 2 training module (latent diffusion)
Single-token VAE: z [1, B, 256]
Multi-GPU (DDP) compatible
"""

import torch
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
    """
    def __init__(
        self,
        # sub-configs (Hydra instantiate)
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
        # misc
        dataset_name: str = "sign",
        evaluator=None,
        save_every_n_epochs: float = 1e9,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # --- Text Encoder (frozen) ---
        self.text_encoder = instantiate(text_encoder)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # --- Noise / Sample Schedulers ---
        self.noise_scheduler = instantiate(noise_scheduler)
        self.sample_scheduler = instantiate(sample_scheduler)

        # --- Denoiser (trainable) ---
        self.denoiser = LatentSignDenoiser(
            **(denoiser if isinstance(denoiser, dict) else {}),
        )

        # --- VAE (frozen, loaded from checkpoint) ---
        self.vae = self._load_vae(vae_checkpoint, vae_config)

        # --- CFG: null text embedding ---
        self.text_replace_prob = text_replace_prob
        self._null_emb = None

        # EMA placeholder (not used by default)
        self.ema_denoiser = None

    def _load_vae(self, ckpt_path, vae_config):
        """Load pre-trained VAE and freeze."""
        if vae_config is None:
            vae_config = {}
        vae = MambaVae(**vae_config)
        if ckpt_path:
            print(f"[LatentSignT2M] Loading VAE from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            # strip 'vae.' prefix if present
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
                self._null_emb = out["text_emb"].detach()  # [1, 512]
        return self._null_emb

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
            z, _ = self.vae.encode(motion, lengths_list)  # [1, B, 256]
        z = z.squeeze(0).unsqueeze(1)  # [B, 1, 256]

        # Text encoding
        with torch.no_grad():
            text_out = self.text_encoder(text, self.device)
            text_emb = text_out["text_emb"].float()  # [B, 512]

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

        # Loss (sample prediction)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'sample':
            target = z
        elif pred_type == 'epsilon':
            target = noise
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        loss = F.mse_loss(z_pred, target)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        motion = batch['motion']
        text = batch['text']
        lengths = batch['motion_len']
        lengths_list = lengths.tolist()

        with torch.no_grad():
            z, _ = self.vae.encode(motion, lengths_list)
        z = z.squeeze(0).unsqueeze(1)

        with torch.no_grad():
            text_out = self.text_encoder(text, self.device)
            text_emb = text_out["text_emb"].float()

        noise = torch.randn_like(z)
        B = z.shape[0]
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device)
        z_noisy = self.noise_scheduler.add_noise(z, noise, t)
        z_pred = self.denoiser(z_noisy, t, text_emb)

        pred_type = self.noise_scheduler.config.prediction_type
        target = z if pred_type == 'sample' else noise
        loss = F.mse_loss(z_pred, target)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

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

        # Init noise
        z = torch.randn(B, 1, self.denoiser.latent_dim, device=device)
        z = z * self.sample_scheduler.init_noise_sigma

        self.sample_scheduler.set_timesteps(self.hparams.step_num)
        pred_type = self.noise_scheduler.config.prediction_type

        for t in self.sample_scheduler.timesteps.to(device):
            z_input = z.repeat(2, 1, 1)  # [2B, 1, 256]
            output = self.denoiser(z_input, t.expand(2 * B), text_emb)

            if pred_type == 'sample':
                cond_x0, uncond_x0 = output.chunk(2)
                alpha_prod_t = self.sample_scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                cond_eps = (z - alpha_prod_t ** 0.5 * cond_x0) / beta_prod_t ** 0.5
                uncond_eps = (z - alpha_prod_t ** 0.5 * uncond_x0) / beta_prod_t ** 0.5
            else:
                cond_eps, uncond_eps = output.chunk(2)

            pred_noise = uncond_eps + self.hparams.guidance_scale * (cond_eps - uncond_eps)
            z = self.sample_scheduler.step(pred_noise, t, z).prev_sample.float()

        # Decode
        z_dec = z.squeeze(1).unsqueeze(0)  # [B,1,256] → [1,B,256]
        motion = self.vae.decode(z_dec, lengths)
        return motion

    # =========================================================================
    # Optimizer (Hydra)
    # =========================================================================
    def configure_optimizers(self):
        opt_cfg = self.hparams.optimizer
        opt = instantiate(opt_cfg, params=self.denoiser.parameters())
        ret = {"optimizer": opt}
        if self.hparams.lr_scheduler is not None:
            sched = instantiate(self.hparams.lr_scheduler, optimizer=opt)
            ret["lr_scheduler"] = {"scheduler": sched, "interval": "epoch"}
        return ret
