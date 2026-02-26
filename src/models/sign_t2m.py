"""
SignMotionGeneration — Motion-space direct diffusion for Sign Language T2M
Based on Light-T2M's LightMotionGeneration

Training: noise → motion [B, T, D] directly
Inference: random noise → denoise → motion [B, T, D]
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

        # text_encoder 각자의 freeze 설정을 존중 (덮어쓰지 않음)
        # - CLIP: __init__에서 전체 freeze 완료
        # - MBartTextEncoder: encoder만 freeze, proj(1024->512)는 trainable 유지

        # Schedulers
        self.noise_scheduler = noise_scheduler
        if sample_scheduler is False:
            self.sample_scheduler = noise_scheduler
            self.sample_scheduler.set_timesteps(1000)
        else:
            self.sample_scheduler = sample_scheduler
            self.sample_scheduler.set_timesteps(step_num)

        # Verify prediction_type
        pt = self.noise_scheduler.config.prediction_type
        print(f"[SignMotionGeneration] noise_scheduler prediction_type = '{pt}'")
        assert pt in ("epsilon", "sample", "v_prediction"), (
            f"prediction_type must be 'epsilon'/'sample'/'v_prediction', got '{pt}'"
        )

        # EMA
        self.ema_denoiser = None
        if isinstance(ema, dict) and ema.get("use_ema", False):
            self.ema_denoiser = EMAModel(
                self.denoiser,
                decay=ema.get("ema_decay", 0.999),
            )

    # =========================================================================
    # Training
    # =========================================================================
    def _get_target(self, motion, noise, timestep, prediction_type):
        """prediction_type에 따른 학습 타겟 반환"""
        if prediction_type == "epsilon":
            return noise
        elif prediction_type == "sample":
            return motion
        elif prediction_type == "v_prediction":
            return self.noise_scheduler.get_velocity(motion, noise, timestep)
        else:
            raise ValueError(f"{prediction_type} not supported!")

    def _step_network(self, batch):
        motion = batch["motion"]      # [B, T, D]
        length = batch["motion_len"]  # [B]
        text   = batch["text"]        # List[str]
        srcs   = batch.get("src")     # List[str] | None

        # CFG: 일부 텍스트를 빈 문자열로 교체
        if self.hparams.text_replace_prob > 0:
            text = [
                "" if torch.rand(1).item() < self.hparams.text_replace_prob else t
                for t in text
            ]

        with torch.no_grad():
            text_embed = self.text_encoder(text, self.device, srcs=srcs)

        B = motion.size(0)
        timestep = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=motion.device,
        ).long()

        padding_mask = lengths_to_mask(length, self.device)
        noise = torch.randn_like(motion)
        x_t = self.noise_scheduler.add_noise(motion, noise, timestep)

        output = self.denoiser(x_t, padding_mask, timestep, text_embed)

        prediction_type = self.noise_scheduler.config.prediction_type
        target = self._get_target(motion, noise, timestep, prediction_type)

        loss_raw = F.mse_loss(output, target, reduction="none")  # [B, T, D]

        # Part-weighted loss: uniform per-dim mean (equal gradient per dim)
        # hand_loss_weight < 1 → body에 상대적으로 더 많은 gradient
        # hand_loss_weight = 1 → 모든 dim 동일 gradient (기본)
        # hand_loss_weight > 1 → hand에 상대적으로 더 많은 gradient
        hand_weight = self.hparams.hand_loss_weight
        if hand_weight != 1.0:
            D = loss_raw.shape[-1]
            weight = torch.ones(D, device=loss_raw.device)
            if D == 120:
                weight[30:75] = hand_weight
                weight[75:120] = hand_weight
            elif D == 133:
                weight[43:88] = hand_weight
                weight[88:133] = hand_weight
            elif D == 360:
                weight[90:225] = hand_weight
                weight[225:360] = hand_weight
            loss_raw = loss_raw * weight

        loss = loss_raw[padding_mask].mean()

        # Velocity loss
        vel_weight = self.hparams.vel_loss_weight
        if vel_weight > 0:
            vel_pred   = output[:, 1:] - output[:, :-1]
            vel_target = target[:, 1:] - target[:, :-1]
            vel_mask   = padding_mask[:, 1:] & padding_mask[:, :-1]
            vel_loss   = F.mse_loss(vel_pred, vel_target, reduction="none")[vel_mask].mean()
            loss = loss + vel_weight * vel_loss
            self.log("train/vel_loss", vel_loss.detach(), prog_bar=True, on_step=False, on_epoch=True)

        # Per-part monitoring (no_grad)
        with torch.no_grad():
            D = output.shape[-1]
            raw_uw = F.mse_loss(output, target, reduction="none")
            if D == 528:
                body_idx, hand_idx = slice(0, 42), slice(42, 132)
            elif D == 360:
                body_idx, hand_idx = slice(0, 90), slice(90, 360)
            elif D == 133:
                body_idx, hand_idx = slice(0, 43), slice(43, 133)
            else:
                body_idx, hand_idx = slice(0, 30), slice(30, D)

            m = padding_mask.unsqueeze(-1)
            body_loss = raw_uw[..., body_idx][m.expand_as(raw_uw[..., body_idx])].mean()
            hand_loss = raw_uw[..., hand_idx][m.expand_as(raw_uw[..., hand_idx])].mean()
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
        text   = batch["text"]
        srcs   = batch.get("src")

        with torch.no_grad():
            text_embed = self.text_encoder(text, self.device, srcs=srcs)

        timestep = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (motion.size(0),), device=motion.device,
        ).long()

        padding_mask = lengths_to_mask(length, self.device)
        noise = torch.randn_like(motion)
        x_t = self.noise_scheduler.add_noise(motion, noise, timestep)
        output = self.denoiser(x_t, padding_mask, timestep, text_embed)

        prediction_type = self.noise_scheduler.config.prediction_type
        # 버그 수정: v_prediction도 올바른 타겟 사용 (기존 else: target=noise 제거)
        target = self._get_target(motion, noise, timestep, prediction_type)

        loss_raw = F.mse_loss(output, target, reduction="none")
        loss = loss_raw[padding_mask].mean()
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # Per-part val losses
        D = loss_raw.shape[-1]
        if D == 528:
            body_idx, hand_idx = slice(0, 42), slice(42, 132)
        elif D == 360:
            body_idx, hand_idx = slice(0, 90), slice(90, 360)
        elif D == 133:
            body_idx, hand_idx = slice(0, 43), slice(43, 133)
        else:
            body_idx, hand_idx = slice(0, 30), slice(30, D)

        m = padding_mask.unsqueeze(-1)
        body_loss = loss_raw[..., body_idx][m.expand_as(loss_raw[..., body_idx])].mean()
        hand_loss = loss_raw[..., hand_idx][m.expand_as(loss_raw[..., hand_idx])].mean()
        self.log("val/body_loss", body_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/hand_loss", hand_loss, prog_bar=True, on_epoch=True, sync_dist=True)

    # =========================================================================
    # Inference
    # =========================================================================
    @torch.no_grad()
    def sample_motion(self, gt_motion, length, text, srcs=None):
        """Generate motion from text via iterative denoising"""
        B, L, D = gt_motion.shape

        # CFG: cond + uncond 텍스트 concat
        repeated_text = list(text) + [""] * B
        srcs_rep = list(srcs) + [''] * B if srcs is not None else None
        text_embed = self.text_encoder(repeated_text, self.device, srcs=srcs_rep)

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
        """x0 prediction → epsilon (for CFG in epsilon space)"""
        alpha_prod_t = self.sample_scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        cond_eps   = (x_t - alpha_prod_t ** 0.5 * cond_x0)   / beta_prod_t ** 0.5
        uncond_eps = (x_t - alpha_prod_t ** 0.5 * uncond_x0) / beta_prod_t ** 0.5
        return cond_eps, uncond_eps

    @torch.no_grad()
    def generate(self, text, lengths, srcs=None):
        """Public API: text + lengths → motion"""
        B = len(text)
        T = max(lengths)
        D = self.denoiser.motion_dim
        dummy = torch.zeros(B, T, D, device=self.device)
        length_tensor = torch.tensor(lengths, device=self.device)
        return self.sample_motion(dummy, length_tensor, text, srcs=srcs)

    # =========================================================================
    # Checkpoint
    # =========================================================================
    def on_save_checkpoint(self, checkpoint):
        """Remove frozen text encoder weights from checkpoint"""
        remove_keys = [k for k in checkpoint["state_dict"] if "text_encoder" in k]
        for k in remove_keys:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint):
        """Handle torch.compile key prefix"""
        keys_list = list(checkpoint["state_dict"].keys())
        for key in keys_list:
            if "_orig_mod." in key:
                new_key = key.replace("_orig_mod.", "")
                checkpoint["state_dict"][new_key] = checkpoint["state_dict"][key]
                del checkpoint["state_dict"][key]

    # =========================================================================
    # Optimizer
    # =========================================================================
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters())
        )
        if self.hparams.lr_scheduler is None:
            return optimizer
        scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }