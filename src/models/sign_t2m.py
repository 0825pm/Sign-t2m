"""
SignMotionGeneration — Motion-space direct diffusion for Sign Language T2M
Based on Light-T2M's LightMotionGeneration (no VAE, no HumanML3D evaluator)

Training: noise → motion [B, T, 120] directly
Inference: random noise → denoise → motion [B, T, 120]
"""

import os
import torch
import torch.nn.functional as F
import lightning.pytorch as L

from diffusers import UniPCMultistepScheduler, DDPMScheduler

from .utils.utils import lengths_to_mask
from .nets.ema import EMAModel


class SignMotionGeneration(L.LightningModule):
    def __init__(
        self,
        text_encoder,
        denoiser,
        noise_scheduler,
        sample_scheduler,
        text_replace_prob,
        guidance_scale,
        dataset_name,
        optimizer,
        ema=False,
        lr_scheduler=None,
        debug=False,
        ocpm=False,
        step_num=10,
        evaluator=None,
        save_every_n_epochs=1e9,
        compile=False,
        hand_loss_weight=1.0,
        vel_loss_weight=0.0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["text_encoder", "denoiser"])
        self.text_encoder = text_encoder
        self.denoiser = denoiser

        # Freeze text encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Schedulers
        self.noise_scheduler = noise_scheduler
        if sample_scheduler is False:
            self.sample_scheduler = noise_scheduler
            self.sample_scheduler.set_timesteps(1000)
        else:
            self.sample_scheduler = sample_scheduler
            self.sample_scheduler.set_timesteps(step_num)

        # ⚠️ Verify prediction_type at startup
        pt = self.noise_scheduler.config.prediction_type
        print(f"[SignMotionGeneration] noise_scheduler prediction_type = '{pt}'")
        assert pt in ("epsilon", "sample", "v_prediction"), (
            f"prediction_type must be 'epsilon'/'sample'/'v_prediction', got '{pt}'. "
            f"Check configs/model/noise_scheduler/DDPM_ori.yaml"
        )

        # EMA
        if isinstance(ema, dict) and ema.get("use_ema", False):
            self.ema_denoiser = EMAModel(self.denoiser, decay=ema["ema_decay"])
            self.ema_denoiser.set(self.denoiser)
        elif hasattr(ema, "use_ema") and ema.use_ema:
            self.ema_denoiser = EMAModel(self.denoiser, decay=ema.ema_decay)
            self.ema_denoiser.set(self.denoiser)
        else:
            self.ema_denoiser = None

        num_params = sum(x.numel() for x in self.denoiser.parameters() if x.requires_grad)
        print(f"[SignMotionGeneration] Trainable params: {num_params / 1e6:.2f}M")

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.denoiser.parameters())
        if self.hparams.lr_scheduler is not None:
            lr_scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
            }
        return optimizer

    # =========================================================================
    # Training
    # =========================================================================
    def _replace_text_with_null(self, text_list, prob):
        """CFG: randomly replace text with empty string"""
        import random
        return [("" if random.random() < prob else t) for t in text_list]

    def _step_network(self, batch):
        motion = batch["motion"]       # [B, T, 120]
        length = batch["motion_len"]   # [B]
        text = batch["text"]           # List[str]

        # CFG: random null text
        text = self._replace_text_with_null(text, self.hparams.text_replace_prob)

        with torch.no_grad():
            text_embed = self.text_encoder(text, self.device)

        # Diffusion forward
        timestep = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (motion.size(0),), device=motion.device,
        ).long()

        padding_mask = lengths_to_mask(length, self.device)
        noise = torch.randn_like(motion)
        x_t = self.noise_scheduler.add_noise(motion, noise, timestep)

        output = self.denoiser(x_t, padding_mask, timestep, text_embed)

        # Loss
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = motion
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(motion, noise, timestep)
        else:
            raise ValueError(f"{prediction_type} not supported!")

        loss_raw = F.mse_loss(output, target, reduction="none")  # [B, T, 120]

        # Part-weighted loss: emphasize hands
        hand_weight = getattr(self.hparams, 'hand_loss_weight', 1.0)
        if hand_weight != 1.0:
            D = loss_raw.shape[-1]
            weight = torch.ones(D, device=loss_raw.device)
            if D == 120:
                # 120D: body[0:30] + lhand[30:75] + rhand[75:120]
                weight[30:75] = hand_weight
                weight[75:120] = hand_weight
            elif D == 528:
                # 528D: 44 joints = 4 spine + 10 body + 15 lhand + 15 rhand
                # positions[0:132]: lhand[42:87], rhand[87:132]
                # velocities[132:264]: lhand[174:219], rhand[219:264]
                # 6D_rot[264:528]: lhand[348:438], rhand[438:528]
                for s, e in [(42,87), (87,132),       # pos lhand, rhand
                             (174,219), (219,264),     # vel lhand, rhand
                             (348,438), (438,528)]:    # rot lhand, rhand
                    weight[s:e] = hand_weight
            loss_raw = loss_raw * weight

        loss = loss_raw[padding_mask].mean()

        # Velocity loss: temporal smoothness (diff along time axis)
        vel_weight = getattr(self.hparams, 'vel_loss_weight', 0.0)
        if vel_weight > 0:
            # Velocity = frame-to-frame difference
            vel_pred = output[:, 1:] - output[:, :-1]   # [B, T-1, 120]
            vel_target = target[:, 1:] - target[:, :-1]
            vel_mask = padding_mask[:, 1:] & padding_mask[:, :-1]  # both frames valid
            vel_loss = F.mse_loss(vel_pred, vel_target, reduction="none")

            # Also apply hand weighting to velocity
            if hand_weight != 1.0:
                vel_loss = vel_loss * weight

            vel_loss = vel_loss[vel_mask].mean()
            loss = loss + vel_weight * vel_loss
            self.log("train/vel_loss", vel_loss.detach(), prog_bar=True, on_step=False, on_epoch=True)

        # Log per-part losses for monitoring
        with torch.no_grad():
            D = output.shape[-1]
            raw_unweighted = F.mse_loss(output, target, reduction="none")
            if D == 528:
                # 528D positions: spine+body[0:42], lhand+rhand[42:132]
                body_idx = slice(0, 42)
                hand_idx = slice(42, 132)
            else:
                # 120D: body[0:30], hand[30:120]
                body_idx = slice(0, 30)
                hand_idx = slice(30, D)
            body_loss = raw_unweighted[..., body_idx][padding_mask[..., None].expand_as(raw_unweighted[..., body_idx])].mean()
            hand_loss = raw_unweighted[..., hand_idx][padding_mask[..., None].expand_as(raw_unweighted[..., hand_idx])].mean()
            self.log("train/body_loss", body_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train/hand_loss", hand_loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        losses = self._step_network(batch)
        self.log("train/loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True)
        return losses["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_denoiser is not None:
            if self.global_step <= self.hparams.ema.ema_start:
                self.ema_denoiser.set(self.denoiser)
            else:
                self.ema_denoiser.update(self.denoiser)

    def on_train_epoch_end(self):
        if (
            self.current_epoch > 0
            and self.current_epoch % self.hparams.save_every_n_epochs == 0
        ):
            save_path = os.path.join(
                self.hparams.get("ckpt_path", "checkpoints"),
                f"epoch-{self.current_epoch}.ckpt",
            )
            self.trainer.save_checkpoint(save_path)

    # =========================================================================
    # Validation
    # =========================================================================
    def validation_step(self, batch, batch_idx):
        motion = batch["motion"]
        length = batch["motion_len"]
        text = batch["text"]

        with torch.no_grad():
            text_embed = self.text_encoder(text, self.device)

        timestep = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (motion.size(0),), device=motion.device,
        ).long()

        padding_mask = lengths_to_mask(length, self.device)
        noise = torch.randn_like(motion)
        x_t = self.noise_scheduler.add_noise(motion, noise, timestep)
        output = self.denoiser(x_t, padding_mask, timestep, text_embed)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = motion
        else:
            target = noise

        loss_raw = F.mse_loss(output, target, reduction="none")
        loss = loss_raw[padding_mask].mean()
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Per-part val losses
        D = loss_raw.shape[-1]
        if D == 528:
            body_idx = slice(0, 42)
            hand_idx = slice(42, 132)
        else:
            body_idx = slice(0, 30)
            hand_idx = slice(30, D)
        body_loss = loss_raw[..., body_idx][padding_mask[..., None].expand_as(loss_raw[..., body_idx])].mean()
        hand_loss = loss_raw[..., hand_idx][padding_mask[..., None].expand_as(loss_raw[..., hand_idx])].mean()
        self.log("val/body_loss", body_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/hand_loss", hand_loss, prog_bar=True, on_epoch=True, sync_dist=True)

    # =========================================================================
    # Inference
    # =========================================================================
    @torch.no_grad()
    def sample_motion(self, gt_motion, length, text):
        """Generate motion from text via iterative denoising"""
        B, L, D = gt_motion.shape

        # CFG: concat cond + uncond text
        repeated_text = list(text) + [""] * B
        text_embed = self.text_encoder(repeated_text, self.device)

        padding_mask = lengths_to_mask(length, self.device)
        pred_motion = torch.randn_like(gt_motion) * self.sample_scheduler.init_noise_sigma

        denoiser = self.ema_denoiser.model if self.ema_denoiser is not None else self.denoiser
        prediction_type = self.noise_scheduler.config.prediction_type

        self.sample_scheduler.set_timesteps(self.hparams.step_num)
        for t in self.sample_scheduler.timesteps.to(self.device):
            output = denoiser(
                pred_motion.repeat(2, 1, 1),
                padding_mask.repeat(2, 1),
                t.repeat(2 * B),
                text_embed,
            )

            if prediction_type == "epsilon":
                cond_eps, uncond_eps = output.chunk(2)
                pred_noise = uncond_eps + self.hparams.guidance_scale * (cond_eps - uncond_eps)
                pred_motion = self.sample_scheduler.step(
                    pred_noise, t, pred_motion
                ).prev_sample.float()

            elif prediction_type == "v_prediction":
                cond_v, uncond_v = output.chunk(2)
                pred_v = uncond_v + self.hparams.guidance_scale * (cond_v - uncond_v)
                pred_motion = self.sample_scheduler.step(
                    pred_v, t, pred_motion
                ).prev_sample.float()

            elif prediction_type == "sample":
                cond_x0, uncond_x0 = output.chunk(2)
                cond_eps, uncond_eps = self._obtain_eps_from_x0(cond_x0, uncond_x0, t, pred_motion)
                pred_noise = uncond_eps + self.hparams.guidance_scale * (cond_eps - uncond_eps)
                pred_motion = self.sample_scheduler.step(
                    pred_noise, t, pred_motion
                ).prev_sample.float()

            else:
                raise ValueError(f"{prediction_type} not supported!")

            pred_motion[~padding_mask] = 0

        return pred_motion

    def _obtain_eps_from_x0(self, cond_x0, uncond_x0, timestep, x_t):
        """Convert x0 prediction to epsilon for CFG"""
        alpha_prod_t = self.sample_scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        cond_eps = (x_t - alpha_prod_t**0.5 * cond_x0) / beta_prod_t**0.5
        uncond_eps = (x_t - alpha_prod_t**0.5 * uncond_x0) / beta_prod_t**0.5
        return cond_eps, uncond_eps

    @torch.no_grad()
    def generate(self, text, lengths):
        """Public API: text + lengths → motion"""
        B = len(text)
        T = max(lengths)
        D = self.denoiser.motion_dim
        dummy = torch.zeros(B, T, D, device=self.device)
        length_tensor = torch.tensor(lengths, device=self.device)
        return self.sample_motion(dummy, length_tensor, text)

    # =========================================================================
    # Checkpoint
    # =========================================================================
    def on_save_checkpoint(self, checkpoint):
        """Remove frozen text encoder from checkpoint"""
        remove_keys = [k for k in checkpoint["state_dict"] if "text_encoder" in k]
        for k in remove_keys:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint):
        """Handle torch.compile key prefix"""
        keys_list = list(checkpoint["state_dict"].keys())
        for key in keys_list:
            if "orig_mod." in key:
                new_key = key.replace("_orig_mod.", "")
                checkpoint["state_dict"][new_key] = checkpoint["state_dict"][key]
                del checkpoint["state_dict"][key]