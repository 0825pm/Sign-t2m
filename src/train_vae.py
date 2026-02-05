"""
train_vae.py — Stage 1: Train MambaVae (single token)
Multi-GPU DDP compatible

Usage:
    # Single GPU
    python src/train_vae.py --devices 0

    # Multi GPU
    python src/train_vae.py --devices 0,1

    # All GPUs
    python src/train_vae.py --devices -1
"""

import os
import sys
import argparse
import datetime
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.distributions import Normal

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.sign_vae import MambaVae
from src.data.sign_datamodule import SignDataModule


class SignVAEModule(LightningModule):
    def __init__(self, vae: MambaVae, lr=2e-4, kl_weight=1e-5, recon_type='l1_smooth', feature_weight=1.0):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        self.vae = vae

    def _compute_loss(self, batch):
        motion = batch['motion']
        lengths = batch['motion_len']
        lengths_list = lengths.tolist()

        recon, z, dist = self.vae(motion, lengths_list)

        # Reconstruction loss (masked)
        from src.models.sign_vae import lengths_to_mask
        mask = lengths_to_mask(lengths_list, motion.device, max_len=motion.shape[1])
        mask_f = mask.unsqueeze(-1).float()

        if self.hparams.recon_type == 'l1_smooth':
            recon_loss = F.smooth_l1_loss(recon * mask_f, motion * mask_f)
        elif self.hparams.recon_type == 'l1':
            recon_loss = F.l1_loss(recon * mask_f, motion * mask_f)
        else:
            recon_loss = F.mse_loss(recon * mask_f, motion * mask_f)

        # KL loss
        mu, logvar = dist.loc, dist.scale.log() * 2
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Feature consistency loss
        z_mean = z.squeeze(0)  # [B, D]
        feature_loss = (z_mean.mean() - self.vae.mean_mean) ** 2 + \
                       (1.0 / (z_mean.std() + 1e-8) - self.vae.mean_std_inv) ** 2

        loss = recon_loss + self.hparams.kl_weight * kl_loss + self.hparams.feature_weight * feature_loss
        return loss, recon_loss, kl_loss, feature_loss

    def training_step(self, batch, batch_idx):
        loss, recon, kl, feat = self._compute_loss(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/recon', recon)
        self.log('train/kl', kl)
        self.log('train/feat', feat)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, kl, feat = self._compute_loss(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/recon_loss', recon, prog_bar=True)
        self.log('val/kl', kl)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.vae.parameters(), lr=self.hparams.lr, betas=(0.9, 0.99))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--data_root', type=str, default='/home/user/Projects/research/SOKE/data/How2Sign')
    p.add_argument('--csl_root', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily')
    p.add_argument('--phoenix_root', type=str, default='/home/user/Projects/research/SOKE/data/Phoenix_2014T')
    p.add_argument('--dataset_name', type=str, default='how2sign_csl_phoenix')
    p.add_argument('--mean_path', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily/mean_120.pt')
    p.add_argument('--std_path', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily/std_120.pt')
    p.add_argument('--csl_mean_path', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_mean_120.pt')
    p.add_argument('--csl_std_path', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_std_120.pt')
    # Model
    p.add_argument('--nfeats', type=int, default=120)
    p.add_argument('--latent_dim', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--num_groups', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.1)
    # Training
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--kl_weight', type=float, default=1e-5)
    p.add_argument('--feature_weight', type=float, default=1.0)
    p.add_argument('--recon_type', type=str, default='l1_smooth')
    p.add_argument('--max_epochs', type=int, default=2000)
    p.add_argument('--precision', type=str, default='32')
    p.add_argument('--num_workers', type=int, default=8)
    # Multi-GPU
    p.add_argument('--devices', type=str, default='0', help='GPU ids, e.g. "0,1" or "-1" for all')
    p.add_argument('--strategy', type=str, default='auto', help='ddp, auto, etc.')
    # Output
    p.add_argument('--output_dir', type=str, default='experiments/sign_vae')
    p.add_argument('--name', type=str, default='mamba_vae')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Parse devices
    if args.devices == '-1':
        devices = -1  # all GPUs
    elif ',' in args.devices:
        devices = [int(x) for x in args.devices.split(',')]
    else:
        devices = [int(args.devices)]

    # Auto-select strategy
    num_gpus = torch.cuda.device_count() if devices == -1 else len(devices) if isinstance(devices, list) else 1
    strategy = 'auto'

    print(f"[train_vae] GPUs: {devices} (num={num_gpus}), strategy: {strategy}")

    # DataModule
    dm = SignDataModule(
        data_root=args.data_root,
        csl_root=args.csl_root,
        phoenix_root=args.phoenix_root,
        mean_path=args.mean_path,
        std_path=args.std_path,
        csl_mean_path=args.csl_mean_path,
        csl_std_path=args.csl_std_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nfeats=args.nfeats,
        dataset_name=args.dataset_name,
        stage='vae',
    )

    # Model
    vae = MambaVae(
        nfeats=args.nfeats,
        latent_dim=[1, args.latent_dim],
        num_layers=args.num_layers,
        num_groups=args.num_groups,
        dropout=args.dropout,
    )
    module = SignVAEModule(
        vae=vae,
        lr=args.lr,
        kl_weight=args.kl_weight,
        feature_weight=args.feature_weight,
        recon_type=args.recon_type,
    )

    # Callbacks
    ckpt_dir = os.path.join(args.output_dir, args.name, 'checkpoints')
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, monitor='val/loss', mode='min', save_top_k=3, save_last=True,
                        filename='{epoch}-{val/loss:.4f}'),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    # Logger
    logger = TensorBoardLogger(save_dir=args.output_dir, name=args.name)

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=10,
        gradient_clip_val=1.0,
    )

    # Pre-cache data (warm up OS filesystem cache before DDP forks)
    print("[train_vae] Pre-loading data (this avoids DDP I/O deadlock)...")
    dm.setup('fit')
    print(f"[train_vae] Data loaded: train={len(dm.train_dataset)}, val={len(dm.val_dataset)}")

    trainer.fit(module, dm)
    print(f"[train_vae] Done. Best: {ckpt_dir}")


if __name__ == '__main__':
    main()