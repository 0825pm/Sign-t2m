"""
vis_vae_sign.py — MambaVae Reconstruction Visualization
GT vs Reconstruction comparison for Sign-t2m VAE

Usage:
    cd ~/Projects/research/sign-t2m
    python vis_vae_sign.py \
        --checkpoint experiments/sign_vae/mamba_vae/checkpoints/last.ckpt \
        --num_samples 3 \
        --output vis_vae_output
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.sign_vae import MambaVae
from src.data.sign_datamodule import SignDataModule
from src.utils.feats2joints import feats2joints_smplx


# =============================================================================
# Constants
# =============================================================================
SOKE_TOTAL_DIM = 120
POSE_SCALE = 2.0

# SMPLX upper body joint indices (55 joints 중 상체+손만)
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_VALID = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND


# =============================================================================
# Skeleton Visualization (from vis_sign_t2m.py)
# =============================================================================

def get_connections(num_joints):
    upper_body = [
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 13), (13, 16), (16, 18), (18, 20),
        (9, 14), (14, 17), (17, 19), (19, 21),
    ]
    hand_connections = []
    if num_joints >= 55:
        for finger in range(5):
            base = 25 + finger * 3
            if base + 2 < num_joints:
                hand_connections.extend([(20, base), (base, base+1), (base+1, base+2)])
        for finger in range(5):
            base = 40 + finger * 3
            if base + 2 < num_joints:
                hand_connections.extend([(21, base), (base, base+1), (base+1, base+2)])
    return [(i, j) for i, j in upper_body + hand_connections if i < num_joints and j < num_joints]


def normalize_to_root(joints, root_idx=9):
    if len(joints.shape) == 3:
        root = joints[:, root_idx:root_idx+1, :]
    else:
        root = joints[root_idx:root_idx+1, :]
    return joints - root


def get_joint_colors():
    return {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}


def save_comparison_video(gt_joints, recon_joints, save_path, title='', fps=20):
    """Save GT vs Reconstruction comparison video (2-panel)."""
    seqs = [gt_joints, recon_joints]
    panel_titles = ['Ground Truth', 'Reconstruction']

    T = min(seq.shape[0] for seq in seqs)
    J = gt_joints.shape[1]

    root_idx = 9 if J > 21 else 0
    normalized_seqs = [normalize_to_root(seq.copy(), root_idx) for seq in seqs]

    all_joints = np.concatenate(normalized_seqs, axis=0)

    if J >= 55:
        valid_idx = SMPLX_VALID
    else:
        valid_idx = list(range(min(22, J)))

    valid_idx_filtered = [i for i in valid_idx if i < J]
    all_x = all_joints[:, valid_idx_filtered, 0].flatten()
    all_y = all_joints[:, valid_idx_filtered, 1].flatten()

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    max_range = max(x_max - x_min, y_max - y_min) * 1.2
    x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
    x_lim = (x_mid - max_range/2, x_mid + max_range/2)
    y_lim = (y_mid - max_range/2, y_mid + max_range/2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if title:
        fig.suptitle(title, fontsize=10)

    for ax, pt in zip(axes, panel_titles):
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(pt, fontsize=11, fontweight='bold')

    connections = get_connections(J)
    colors = get_joint_colors()
    all_lines, all_scatters = [], []

    for ax in axes:
        lines = []
        for (i, j) in connections:
            if i >= 40 or j >= 40:
                c, lw = colors['rhand'], 1.0
            elif i >= 25 or j >= 25:
                c, lw = colors['lhand'], 1.0
            else:
                c, lw = colors['body'], 1.5
            line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
            lines.append((line, i, j))
        all_lines.append(lines)

        bs = ax.scatter([], [], c=colors['body'], s=10, zorder=5)
        ls = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
        rs = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
        all_scatters.append((bs, ls, rs))

    plt.tight_layout()
    ub_idx = [i for i in SMPLX_UPPER_BODY if i < J]

    def update(frame):
        for seq, lines, scatters in zip(normalized_seqs, all_lines, all_scatters):
            fd = seq[min(frame, len(seq)-1)]
            x, y = fd[:, 0], fd[:, 1]
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            bs, ls, rs = scatters
            bs.set_offsets(np.c_[x[ub_idx], y[ub_idx]])
            if J > 25:
                ls.set_offsets(np.c_[x[25:40], y[25:40]])
            if J > 40:
                rs.set_offsets(np.c_[x[40:55], y[40:55]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(gt_feats, recon_feats, gt_joints=None, recon_joints=None):
    metrics = {}
    metrics['feat_l1'] = float(F.l1_loss(gt_feats, recon_feats))
    metrics['feat_mse'] = float(F.mse_loss(gt_feats, recon_feats))
    metrics['feat_l1_body'] = float(F.l1_loss(gt_feats[..., :30], recon_feats[..., :30]))
    metrics['feat_l1_lhand'] = float(F.l1_loss(gt_feats[..., 30:75], recon_feats[..., 30:75]))
    metrics['feat_l1_rhand'] = float(F.l1_loss(gt_feats[..., 75:120], recon_feats[..., 75:120]))

    if gt_joints is not None and recon_joints is not None:
        diff = (gt_joints - recon_joints)
        per_joint = torch.sqrt((diff ** 2).sum(-1))  # [T, J]
        metrics['mpjpe'] = float(per_joint.mean())
        ub = [i for i in SMPLX_UPPER_BODY if i < gt_joints.shape[1]]
        if ub:
            metrics['mpjpe_body'] = float(per_joint[:, ub].mean())
        lh = [i for i in SMPLX_LHAND if i < gt_joints.shape[1]]
        if lh:
            metrics['mpjpe_lhand'] = float(per_joint[:, lh].mean())
        rh = [i for i in SMPLX_RHAND if i < gt_joints.shape[1]]
        if rh:
            metrics['mpjpe_rhand'] = float(per_joint[:, rh].mean())
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='vis_vae_output')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--split', type=str, default='val',
                        help='val, test, or all (=val+test)')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0')
    # Data paths
    parser.add_argument('--data_root', type=str, default='/home/user/Projects/research/SOKE/data/How2Sign')
    parser.add_argument('--csl_root', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily')
    parser.add_argument('--phoenix_root', type=str, default='/home/user/Projects/research/SOKE/data/Phoenix_2014T')
    parser.add_argument('--mean_path', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily/mean_120.pt')
    parser.add_argument('--std_path', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily/std_120.pt')
    parser.add_argument('--csl_mean_path', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_mean_120.pt')
    parser.add_argument('--csl_std_path', type=str, default='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_std_120.pt')
    # VAE config
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'vae_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("MambaVae Reconstruction Visualization")
    print("=" * 60)

    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1/3] Loading data...")
    dm = SignDataModule(
        data_root=args.data_root,
        csl_root=args.csl_root,
        phoenix_root=args.phoenix_root,
        mean_path=args.mean_path,
        std_path=args.std_path,
        csl_mean_path=args.csl_mean_path,
        csl_std_path=args.csl_std_path,
        batch_size=1,
        num_workers=0,
        nfeats=120,
        dataset_name='how2sign_csl_phoenix',
        stage='vae',
    )
    dm.setup(None)  # load fit + test

    mean = dm.mean.to(device)
    std = dm.std.to(device)
    csl_mean = dm.csl_mean.to(device) if hasattr(dm, 'csl_mean') else mean
    csl_std = dm.csl_std.to(device) if hasattr(dm, 'csl_std') else std

    # =========================================================================
    # 2. Load VAE
    # =========================================================================
    print("\n[2/3] Loading VAE...")
    vae = MambaVae(
        nfeats=120,
        latent_dim=[1, args.latent_dim],
        num_layers=args.num_layers,
    )

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    cleaned = {}
    for k, v in state.items():
        k2 = k.replace('vae.', '') if k.startswith('vae.') else k
        cleaned[k2] = v
    vae.load_state_dict(cleaned, strict=False)

    vae.eval().to(device)

    epoch_info = ckpt.get('epoch', '?')
    step_info = ckpt.get('global_step', '?')
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Epoch: {epoch_info}, Step: {step_info}")
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"  Params: {total_params/1e6:.2f}M")

    # =========================================================================
    # 3. Reconstruct per split × dataset
    # =========================================================================
    splits = ['val', 'test'] if args.split == 'all' else [args.split]
    DATASETS = ['h2s', 'csl', 'phoenix']
    global_metrics = []  # all metrics across splits/datasets

    for split in splits:
        if split == 'val':
            dataloader = dm.val_dataloader()
            dataset = dm.val_dataset
        elif split == 'test':
            dataloader = dm.test_dataloader()
            dataset = dm.test_dataset
        else:
            dataloader = dm.train_dataloader()
            dataset = dm.train_dataset

        print(f"\n{'=' * 60}")
        print(f"Split: {split} ({len(dataset)} samples)")
        print(f"{'=' * 60}")

        # Collect samples grouped by dataset source
        src_batches = {ds: [] for ds in DATASETS}
        for iter_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            # Read src directly from dataset
            src = dataset.all_data[iter_idx].get('src', 'h2s') if hasattr(dataset, 'all_data') else 'h2s'
            if src == 'how2sign':
                src = 'h2s'
            if src not in src_batches:
                src = 'h2s'
            if len(src_batches[src]) < args.num_samples:
                src_batches[src].append((batch, src))
            if all(len(v) >= args.num_samples for v in src_batches.values()):
                break

        for ds_name in DATASETS:
            items = src_batches[ds_name]
            if not items:
                print(f"\n  [{split}/{ds_name}] No samples found, skipping.")
                continue

            output_dir = os.path.join(output_root, f'{split}_{ds_name}')
            os.makedirs(output_dir, exist_ok=True)
            all_metrics = []

            print(f"\n  [{split}/{ds_name}] Reconstructing {len(items)} samples...")

            for idx, (batch, src) in enumerate(items):
                motion = batch['motion'].to(device)
                length = batch['motion_len']
                text = batch['text'][0] if 'text' in batch else ''
                name = batch['name'][0] if 'name' in batch else f'sample_{idx}'

                length_val = int(length[0])

                with torch.no_grad():
                    recon, z, dist = vae(motion, [length_val])

                gt_feats = motion[0, :length_val, :]
                recon_feats = recon[0, :length_val, :]

                if src == 'csl':
                    m, s = csl_mean, csl_std
                else:
                    m, s = mean, std

                try:
                    _, gt_joints_full = feats2joints_smplx(gt_feats.unsqueeze(0), m, s)
                    _, recon_joints_full = feats2joints_smplx(recon_feats.unsqueeze(0), m, s)

                    gt_joints_full = gt_joints_full[0]
                    recon_joints_full = recon_joints_full[0]

                    if gt_joints_full.shape[1] > 55:
                        gt_joints_55 = gt_joints_full[:, :55, :]
                        recon_joints_55 = recon_joints_full[:, :55, :]
                    else:
                        gt_joints_55 = gt_joints_full
                        recon_joints_55 = recon_joints_full

                    gt_j_np = gt_joints_55.cpu().numpy()
                    recon_j_np = recon_joints_55.cpu().numpy()
                    has_joints = True
                except Exception as e:
                    print(f"    Warning: feats2joints failed ({e})")
                    has_joints = False

                metrics = compute_metrics(gt_feats, recon_feats,
                                          gt_joints_55 if has_joints else None,
                                          recon_joints_55 if has_joints else None)
                metrics['name'] = name
                metrics['text'] = text
                metrics['src'] = src
                metrics['split'] = split
                metrics['length'] = length_val
                metrics['z_mean'] = float(z.mean())
                metrics['z_std'] = float(z.std())
                all_metrics.append(metrics)

                print(f"    [{idx+1}/{len(items)}] {name} (T={length_val})")
                print(f"      L1: {metrics['feat_l1']:.4f} (body={metrics['feat_l1_body']:.4f}, "
                      f"lhand={metrics['feat_l1_lhand']:.4f}, rhand={metrics['feat_l1_rhand']:.4f})")
                if 'mpjpe' in metrics:
                    print(f"      MPJPE: {metrics['mpjpe']:.4f}")

                if has_joints:
                    safe_name = name[:30].replace('/', '_').replace('\\', '_')
                    video_path = os.path.join(output_dir, f'{idx:03d}_{safe_name}.mp4')
                    title_str = f'{name} [{src}/{split}] (L1={metrics["feat_l1"]:.4f})\n{text[:60]}'
                    save_comparison_video(gt_j_np, recon_j_np, video_path, title_str, args.fps)

            if all_metrics:
                print(f"\n  --- {split}/{ds_name} Summary ({len(all_metrics)} samples) ---")
                print(f"    Avg L1: {np.mean([m['feat_l1'] for m in all_metrics]):.4f}")
                if 'mpjpe' in all_metrics[0]:
                    print(f"    Avg MPJPE: {np.mean([m['mpjpe'] for m in all_metrics]):.4f}")

                json_path = os.path.join(output_dir, 'metrics.json')
                with open(json_path, 'w') as f:
                    json.dump({
                        'checkpoint': args.checkpoint,
                        'epoch': str(epoch_info),
                        'split': split,
                        'dataset': ds_name,
                        'num_samples': len(all_metrics),
                        'avg_l1': float(np.mean([m['feat_l1'] for m in all_metrics])),
                        'avg_mpjpe': float(np.mean([m['mpjpe'] for m in all_metrics])) if 'mpjpe' in all_metrics[0] else None,
                        'samples': all_metrics,
                    }, f, indent=2, default=str)

            global_metrics.extend(all_metrics)

    # =========================================================================
    # Global Summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Global Summary")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(global_metrics)}")
    if global_metrics:
        print(f"Avg L1:       {np.mean([m['feat_l1'] for m in global_metrics]):.4f}")
        print(f"Avg L1 Body:  {np.mean([m['feat_l1_body'] for m in global_metrics]):.4f}")
        print(f"Avg L1 LHand: {np.mean([m['feat_l1_lhand'] for m in global_metrics]):.4f}")
        print(f"Avg L1 RHand: {np.mean([m['feat_l1_rhand'] for m in global_metrics]):.4f}")
        if 'mpjpe' in global_metrics[0]:
            print(f"Avg MPJPE:    {np.mean([m['mpjpe'] for m in global_metrics]):.4f}")

        # Per-dataset breakdown
        for ds in DATASETS:
            ds_m = [m for m in global_metrics if m['src'] == ds]
            if ds_m:
                print(f"  [{ds:>7s}] L1={np.mean([m['feat_l1'] for m in ds_m]):.4f}"
                      + (f"  MPJPE={np.mean([m['mpjpe'] for m in ds_m]):.4f}" if 'mpjpe' in ds_m[0] else ''))

        # Save global metrics
        json_path = os.path.join(output_root, 'global_metrics.json')
        with open(json_path, 'w') as f:
            json.dump({
                'checkpoint': args.checkpoint,
                'epoch': str(epoch_info),
                'splits': splits,
                'total_samples': len(global_metrics),
                'avg_l1': float(np.mean([m['feat_l1'] for m in global_metrics])),
                'avg_mpjpe': float(np.mean([m['mpjpe'] for m in global_metrics])) if 'mpjpe' in global_metrics[0] else None,
                'samples': global_metrics,
            }, f, indent=2, default=str)
        print(f"\nGlobal metrics saved: {json_path}")

    print(f"Output dir: {output_root}")


if __name__ == '__main__':
    main()