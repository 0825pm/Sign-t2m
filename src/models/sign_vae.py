"""
SignVAE — Lightning module for MLD-style Motion VAE training (Stage 1).

Text-free, motion-only training.
Loss = Reconstruction (L1 smooth) + Velocity (L1 smooth) + β·KL divergence.
"""

import os
import torch
import torch.nn.functional as F
import lightning.pytorch as L

from .utils.utils import lengths_to_mask


class SignVAE(L.LightningModule):
    def __init__(
        self,
        vae,
        optimizer,
        lr_scheduler=None,
        # Loss weights
        lambda_recon: float = 1.0,
        lambda_velocity: float = 0.5,
        lambda_kl: float = 1e-5,
        recon_loss_type: str = "l1_smooth",   # "l1", "l2", "l1_smooth"
        # Part weighting
        hand_loss_weight: float = 2.0,
        pos_weight: float = 1.0,
        vel_weight: float = 0.5,
        rot_weight: float = 0.2,
        # Misc
        save_every_n_epochs: float = 1e9,
        ckpt_path: str = "checkpoints",
        compile: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["vae"])
        self.vae = vae

        # Reconstruction loss function
        if recon_loss_type == "l1":
            self.recon_loss_fn = F.l1_loss
        elif recon_loss_type == "l2":
            self.recon_loss_fn = F.mse_loss
        elif recon_loss_type == "l1_smooth":
            self.recon_loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown recon_loss_type: {recon_loss_type}")

        num_params = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        print(f"[SignVAE] Trainable params: {num_params / 1e6:.2f}M")

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            self.vae = torch.compile(self.vae)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.vae.parameters())
        if self.hparams.lr_scheduler is not None:
            lr_scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
            }
        return optimizer

    # =========================================================================
    # Loss computation
    # =========================================================================
    def _build_part_weight(self, D, device):
        """Build per-feature weight tensor with part-level and hand emphasis."""
        weight = torch.ones(D, device=device)
        hw = self.hparams.hand_loss_weight
        pw = getattr(self.hparams, 'pos_weight', 1.0)
        vw = getattr(self.hparams, 'vel_weight', 0.5)
        rw = getattr(self.hparams, 'rot_weight', 0.2)

        if D == 528:
            # Part-level weighting
            weight[0:132] = pw      # positions
            weight[132:264] = vw    # velocities
            weight[264:528] = rw    # 6D rotations

            # Hand emphasis (on top of part weight)
            if hw > 1.0:
                for s, e in [(42, 87), (87, 132),       # pos hands
                             (174, 219), (219, 264),     # vel hands
                             (348, 438), (438, 528)]:    # rot hands
                    weight[s:e] *= hw
        elif D == 107:
            weight[17:62] = hw    # lhand
            weight[62:107] = hw   # rhand
        elif D == 120:
            weight[30:75] = hw    # lhand
            weight[75:120] = hw   # rhand
        elif D == 133:
            # 133D: root+body_ric[0:43] | lhand_ric[43:88] | rhand_ric[88:133]
            weight[43:88] = hw    # lhand
            weight[88:133] = hw   # rhand
        return weight

    def _compute_loss(self, feats_ref, feats_rst, dist, lengths):
        """
        Compute total VAE loss.

        Args:
            feats_ref: [B, T, D] — ground truth motion
            feats_rst: [B, T, D] — reconstructed motion
            dist:      torch.distributions.Normal — latent distribution
            lengths:   [B] tensor — valid lengths
        """
        device = feats_ref.device
        D = feats_ref.shape[-1]
        mask = lengths_to_mask(lengths, device)  # [B, T]

        # Ensure same temporal length (decoder may differ slightly)
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        feats_ref = feats_ref[:, :min_len]
        feats_rst = feats_rst[:, :min_len]
        mask = mask[:, :min_len]

        part_weight = self._build_part_weight(D, device)

        # --- 1) Reconstruction loss ---
        recon_raw = self.recon_loss_fn(feats_rst, feats_ref, reduction="none")  # [B, T, D]
        recon_raw = recon_raw * part_weight  # part weighting
        # Mask padded frames
        recon_loss = recon_raw[mask].mean()

        # --- 2) Velocity loss (temporal smoothness) ---
        vel_loss = torch.tensor(0.0, device=device)
        if self.hparams.lambda_velocity > 0:
            vel_ref = feats_ref[:, 1:] - feats_ref[:, :-1]
            vel_rst = feats_rst[:, 1:] - feats_rst[:, :-1]
            vel_mask = mask[:, 1:] & mask[:, :-1]
            vel_raw = self.recon_loss_fn(vel_rst, vel_ref, reduction="none")
            vel_raw = vel_raw * part_weight
            if vel_mask.any():
                vel_loss = vel_raw[vel_mask].mean()

        # --- 3) KL divergence loss ---
        kl_loss = torch.tensor(0.0, device=device)
        if dist is not None and self.hparams.lambda_kl > 0:
            mu_ref = torch.zeros_like(dist.loc)
            scale_ref = torch.ones_like(dist.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref, validate_args=False)
            kl_loss = torch.distributions.kl_divergence(dist, dist_ref).mean()
            # Guard against fp16 NaN
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.0, device=device)

        # --- Total ---
        total = (
            self.hparams.lambda_recon * recon_loss
            + self.hparams.lambda_velocity * vel_loss
            + self.hparams.lambda_kl * kl_loss
        )

        # Guard: skip NaN batches
        if torch.isnan(total):
            total = (feats_rst * 0.0).sum()  # zero loss, keeps graph for DDP

        return {
            "loss": total,
            "recon_loss": recon_loss.detach(),
            "vel_loss": vel_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }

    # =========================================================================
    # Training
    # =========================================================================
    def training_step(self, batch, batch_idx):
        motion = batch["motion"]         # [B, T, D]
        lengths = batch["motion_len"]    # [B]

        # VAE forward
        feats_rst, z, dist = self.vae(motion, lengths.tolist())

        # Loss
        losses = self._compute_loss(motion, feats_rst, dist, lengths)

        # Logging
        self.log("train/loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/recon", losses["recon_loss"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/vel", losses["vel_loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/kl", losses["kl_loss"], on_step=False, on_epoch=True, sync_dist=True)

        # Monitor latent statistics
        with torch.no_grad():
            self.log("train/z_mean", z.mean(), on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/z_std", z.std(), on_step=False, on_epoch=True, sync_dist=True)

        return losses["loss"]

    def on_train_epoch_end(self):
        if (
            self.current_epoch == 0
            or self.current_epoch % self.hparams.save_every_n_epochs == 0
        ):
            save_path = os.path.join(
                self.hparams.get("ckpt_path", "checkpoints"),
                f"vae-epoch-{self.current_epoch}.ckpt",
            )
            self.trainer.save_checkpoint(save_path)

    # =========================================================================
    # Validation
    # =========================================================================
    def validation_step(self, batch, batch_idx):
        motion = batch["motion"]
        lengths = batch["motion_len"]

        feats_rst, z, dist = self.vae(motion, lengths.tolist())
        losses = self._compute_loss(motion, feats_rst, dist, lengths)

        self.log("val/loss", losses["loss"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/recon", losses["recon_loss"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/vel", losses["vel_loss"], on_epoch=True, sync_dist=True)
        self.log("val/kl", losses["kl_loss"], on_epoch=True, sync_dist=True)

        # Per-part reconstruction RMSE (for quality monitoring)
        with torch.no_grad():
            min_len = min(motion.shape[1], feats_rst.shape[1])
            diff = (motion[:, :min_len] - feats_rst[:, :min_len])
            mask = lengths_to_mask(lengths, motion.device)[:, :min_len]
            mse = (diff ** 2)

            D = diff.shape[-1]
            if D == 528:
                body_idx = slice(0, 42)
                hand_idx = slice(42, 132)
            elif D == 133:
                body_idx = slice(0, 43)
                hand_idx = slice(43, 133)
            elif D == 107:
                body_idx = slice(0, 17)
                hand_idx = slice(17, 107)
            else:
                body_idx = slice(0, 30)
                hand_idx = slice(30, D)

            body_rmse = mse[..., body_idx][mask.unsqueeze(-1).expand_as(mse[..., body_idx])].mean().sqrt()
            hand_rmse = mse[..., hand_idx][mask.unsqueeze(-1).expand_as(mse[..., hand_idx])].mean().sqrt()

            self.log("val/body_rmse", body_rmse, on_epoch=True, sync_dist=True)
            self.log("val/hand_rmse", hand_rmse, prog_bar=True, on_epoch=True, sync_dist=True)