"""
Light-T2M Text-to-Motion Visualization
Based on SignGPT3 vis_lm_auto.py

Usage:
    cd ~/Projects/research/sign-t2m
    python vis_sign_t2m.py --checkpoint logs/light/runs/XXXX/checkpoints/last.ckpt \
        --output vis_output --num_samples 3
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from types import SimpleNamespace

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# SOKE 133 dims format constants
# =============================================================================
SOKE_BODY_DIM = 30      # upper body (10 joints × 3)
SOKE_LHAND_DIM = 45     # left hand (15 joints × 3)
SOKE_RHAND_DIM = 45     # right hand (15 joints × 3)
SOKE_JAW_DIM = 3        # jaw (1 joint × 3)
SOKE_EXPR_DIM = 10      # expression (10 dims)
SOKE_TOTAL_DIM = 120
POSE_SCALE = 2.0


# =============================================================================
# Skeleton Visualization Utilities
# =============================================================================

def get_connections(num_joints):
    """Get skeleton connections for upper body + hands."""
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
                hand_connections.extend([
                    (20, base), (base, base + 1), (base + 1, base + 2)
                ])
        for finger in range(5):
            base = 40 + finger * 3
            if base + 2 < num_joints:
                hand_connections.extend([
                    (21, base), (base, base + 1), (base + 1, base + 2)
                ])
    
    all_connections = upper_body + hand_connections
    return [(i, j) for i, j in all_connections if i < num_joints and j < num_joints]


def normalize_to_root(joints, root_idx=9):
    """Normalize joints relative to root joint."""
    if len(joints.shape) == 3:
        root = joints[:, root_idx:root_idx+1, :]
    else:
        root = joints[root_idx:root_idx+1, :]
    return joints - root


def get_joint_colors():
    return {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}


# =============================================================================
# Video Saving Functions
# =============================================================================

def save_comparison_video(gt_joints, recon_joints, save_path, title='', fps=20):
    """Save GT vs Reconstruction comparison video (2-panel)."""
    seqs = [gt_joints, recon_joints]
    panel_titles = ['Ground Truth', 'Generated']
    
    T = min(seq.shape[0] for seq in seqs)
    J = gt_joints.shape[1]
    
    root_idx = 9 if J > 21 else 0
    normalized_seqs = [normalize_to_root(seq.copy(), root_idx) for seq in seqs]
    
    all_joints = np.concatenate(normalized_seqs, axis=0)
    
    if J >= 55:
        upper_body_idx = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        hand_idx = list(range(25, min(55, J)))
        valid_idx = upper_body_idx + hand_idx
    else:
        valid_idx = list(range(min(22, J)))
    
    all_x = all_joints[:, valid_idx, 0].flatten()
    all_y = all_joints[:, valid_idx, 1].flatten()
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    max_range = max(x_max - x_min, y_max - y_min) * 1.2
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    
    x_lim = (x_mid - max_range/2, x_mid + max_range/2)
    y_lim = (y_mid - max_range/2, y_mid + max_range/2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if title:
        fig.suptitle(title[:100], fontsize=10)
    
    for ax, panel_title in zip(axes, panel_titles):
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
    
    connections = get_connections(J)
    colors = get_joint_colors()
    
    all_lines = []
    all_scatters = []
    
    for ax in axes:
        lines = []
        for (i, j) in connections:
            if i >= 40 or j >= 40:
                color = colors['rhand']
                lw = 1.0
            elif i >= 25 or j >= 25:
                color = colors['lhand']
                lw = 1.0
            else:
                color = colors['body']
                lw = 1.5
            line, = ax.plot([], [], color=color, linewidth=lw, alpha=0.8)
            lines.append((line, i, j))
        all_lines.append(lines)
        
        body_scatter = ax.scatter([], [], c=colors['body'], s=10, zorder=5)
        lhand_scatter = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
        rhand_scatter = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
        all_scatters.append((body_scatter, lhand_scatter, rhand_scatter))
    
    plt.tight_layout()
    
    upper_body_idx = [i for i in [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] if i < J]
    
    def update(frame):
        for panel_idx, (seq, lines, scatters) in enumerate(zip(normalized_seqs, all_lines, all_scatters)):
            frame_data = seq[min(frame, len(seq)-1)]
            x, y = frame_data[:, 0], frame_data[:, 1]
            
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            
            body_scatter, lhand_scatter, rhand_scatter = scatters
            body_scatter.set_offsets(np.c_[x[upper_body_idx], y[upper_body_idx]])
            
            if J > 25:
                lhand_scatter.set_offsets(np.c_[x[25:40], y[25:40]])
            if J > 40:
                rhand_scatter.set_offsets(np.c_[x[40:55], y[40:55]])
        return []
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(save_path, writer=writer)
    except Exception as e:
        print(f"    FFMpeg error: {e}, trying GIF...")
        gif_path = save_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=min(fps, 10))
    
    plt.close(fig)


def save_three_view_video(gt_joints, recon_joints, save_path, title='', fps=20):
    """Save 3-view (front, side, top) comparison video."""
    T = min(gt_joints.shape[0], recon_joints.shape[0])
    J = gt_joints.shape[1]
    
    root_idx = 9 if J > 21 else 0
    gt_norm = normalize_to_root(gt_joints.copy(), root_idx)
    recon_norm = normalize_to_root(recon_joints.copy(), root_idx)
    
    views = [('Front', 0, 1), ('Side', 2, 1), ('Top', 0, 2)]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if title:
        fig.suptitle(title[:100], fontsize=11)
    
    bounds = {}
    for view_name, xi, yi in views:
        all_data = np.concatenate([gt_norm, recon_norm], axis=0)
        all_x = all_data[:, :, xi].flatten()
        all_y = all_data[:, :, yi].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) * 1.2
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        bounds[view_name] = {
            'xlim': (x_mid - max_range/2, x_mid + max_range/2),
            'ylim': (y_mid - max_range/2, y_mid + max_range/2),
        }
    
    connections = get_connections(J)
    row_titles = ['Ground Truth', 'Generated']
    all_elements = []
    
    for row, (seq, row_title) in enumerate(zip([gt_norm, recon_norm], row_titles)):
        for col, (view_name, xi, yi) in enumerate(views):
            ax = axes[row, col]
            b = bounds[view_name]
            ax.set_xlim(b['xlim'])
            ax.set_ylim(b['ylim'])
            ax.set_aspect('equal')
            ax.axis('off')
            
            if row == 0:
                ax.set_title(view_name, fontsize=10, fontweight='bold')
            if col == 0:
                ax.text(-0.15, 0.5, row_title, transform=ax.transAxes,
                       fontsize=10, fontweight='bold', rotation=90, va='center')
            
            lines = []
            for (i, j) in connections:
                if i >= 40 or j >= 40:
                    color = 'green'
                elif i >= 25 or j >= 25:
                    color = 'red'
                else:
                    color = 'blue'
                line, = ax.plot([], [], color=color, linewidth=1.2, alpha=0.8)
                lines.append((line, i, j))
            
            scatter = ax.scatter([], [], c='black', s=5, zorder=5)
            all_elements.append((seq, lines, scatter, xi, yi))
    
    plt.tight_layout()
    
    def update(frame):
        for (seq, lines, scatter, xi, yi) in all_elements:
            frame_data = seq[min(frame, len(seq)-1)]
            x, y = frame_data[:, xi], frame_data[:, yi]
            
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            
            scatter.set_offsets(np.c_[x, y])
        return []
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    
    try:
        writer = FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(save_path, writer=writer)
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    
    plt.close(fig)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(gt_feats, recon_feats, gt_joints=None, recon_joints=None):
    """Compute reconstruction metrics."""
    metrics = {}
    
    # Feature space metrics
    metrics['feat_mse'] = float(((gt_feats - recon_feats) ** 2).mean())
    metrics['feat_rmse'] = float(np.sqrt(metrics['feat_mse']))
    metrics['feat_l1'] = float(np.abs(gt_feats - recon_feats).mean())
    
    # Part-wise feature metrics (SOKE 133-dim)
    D = gt_feats.shape[-1]
    if D >= 120:
        metrics['feat_rmse_body'] = float(np.sqrt(((gt_feats[:, 0:30] - recon_feats[:, 0:30]) ** 2).mean()))
        metrics['feat_rmse_lhand'] = float(np.sqrt(((gt_feats[:, 30:75] - recon_feats[:, 30:75]) ** 2).mean()))
        metrics['feat_rmse_rhand'] = float(np.sqrt(((gt_feats[:, 75:120] - recon_feats[:, 75:120]) ** 2).mean()))
    if D >= 133:
        metrics['feat_rmse_jaw'] = float(np.sqrt(((gt_feats[:, 120:123] - recon_feats[:, 120:123]) ** 2).mean()))
        metrics['feat_rmse_expr'] = float(np.sqrt(((gt_feats[:, 123:133] - recon_feats[:, 123:133]) ** 2).mean()))
    
    # Joint space metrics
    if gt_joints is not None and recon_joints is not None:
        if not np.isnan(recon_joints).any():
            min_len = min(len(gt_joints), len(recon_joints))
            gt_j = gt_joints[:min_len]
            recon_j = recon_joints[:min_len]
            
            diff = gt_j - recon_j
            dist = np.sqrt((diff ** 2).sum(axis=-1))
            metrics['mpjpe'] = float(dist.mean())
            
            J = gt_j.shape[1]
            if J >= 55:
                body_idx = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                metrics['mpjpe_body'] = float(dist[:, body_idx].mean())
                metrics['mpjpe_lhand'] = float(dist[:, 25:40].mean())
                metrics['mpjpe_rhand'] = float(dist[:, 40:55].mean())
        else:
            metrics['mpjpe'] = float('nan')
    
    return metrics


# =============================================================================
# LM Forward Function (Text → Motion via Diffusion)
# =============================================================================

def lm_forward(model, text, length, device, num_inference_steps=10, guidance_scale=4.0):
    """
    Forward pass through Light-T2M (Text → Motion).
    """
    length_tensor = torch.tensor([length], device=device)
    dummy_motion = torch.zeros(1, length, SOKE_TOTAL_DIM, device=device)
    
    feats_pred = model.sample_motion(dummy_motion, length_tensor, [text])
    return feats_pred


# =============================================================================
# Dataset Source Detection
# =============================================================================

def normalize_source_name(src):
    """Normalize dataset source name."""
    src_lower = src.lower() if src else 'unknown'
    
    if 'how2sign' in src_lower or 'h2s' in src_lower:
        return 'how2sign'
    elif 'csl' in src_lower:
        return 'csl'
    elif 'phoenix' in src_lower:
        return 'phoenix'
    else:
        return src_lower


# =============================================================================
# Main Visualization Function
# =============================================================================

def main():
    # =========================
    # Parse Arguments
    # =========================
    parser = argparse.ArgumentParser(description='Light-T2M Text-to-Motion Visualization')
    parser.add_argument('--num_samples', type=int, default=2,
                       help='Number of samples per dataset per split (default: 2)')
    parser.add_argument('--output', type=str, default='vis_output',
                       help='Output directory (default: vis_output)')
    parser.add_argument('--fps', type=int, default=25,
                       help='Video FPS (default: 25)')
    parser.add_argument('--three_view', action='store_true',
                       help='Generate 3-view videos instead of 2-panel')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint path')
    parser.add_argument('--splits', type=str, default='val',
                       help='Comma-separated splits to process (default: val)')
    parser.add_argument('--datasets', type=str, default='how2sign,csl,phoenix',
                       help='Comma-separated datasets (default: how2sign,csl,phoenix)')
    parser.add_argument('--no_video', action='store_true',
                       help='Skip video generation (metrics only)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                       help='Number of diffusion steps (default: 10)')
    parser.add_argument('--guidance_scale', type=float, default=4.0,
                       help='Classifier-free guidance scale (default: 4.0)')
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Parse splits and datasets
    splits_to_process = [s.strip() for s in args.splits.split(',')]
    datasets_to_process = [d.strip().lower() for d in args.datasets.split(',')]
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================
    # Print Configuration
    # =========================
    print(f"\n{'='*70}")
    print("Light-T2M Text-to-Motion Visualization")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Samples per dataset per split: {args.num_samples}")
    print(f"Splits: {splits_to_process}")
    print(f"Datasets: {datasets_to_process}")
    print(f"Video mode: {'3-view' if args.three_view else '2-panel'}")
    print(f"FPS: {args.fps}")
    print(f"Device: {device}")
    print(f"Feature dim: {SOKE_TOTAL_DIM}")
    
    # =========================
    # Build DataModule
    # =========================
    print("\n[1/3] Loading datamodule...")
    from src.data.sign_datamodule import SignDataModule
    
    datamodule = SignDataModule(
        data_root='/home/user/Projects/research/SOKE/data/How2Sign',
        csl_root='/home/user/Projects/research/SOKE/data/CSL-Daily',
        phoenix_root='/home/user/Projects/research/SOKE/data/Phoenix_2014T',
        mean_path='/home/user/Projects/research/SOKE/data/CSL-Daily/mean_133.pt',
        std_path='/home/user/Projects/research/SOKE/data/CSL-Daily/std_133.pt',
        csl_mean_path='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_mean_133.pt',
        csl_std_path='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_std_133.pt',
        batch_size=1,
        nfeats=SOKE_TOTAL_DIM,
        njoints=55,
        max_motion_length=300,
        min_motion_length=40,
        stage='lm',
    )
    
    # =========================
    # Build Model
    # =========================
    print("[2/3] Loading model...")
    from diffusers import DDPMScheduler, UniPCMultistepScheduler
    from src.models.nets.light_final import LightT2M
    from src.models.nets.text_encoder import CLIP
    from src.models.light_final import LightMotionGeneration
    
    text_encoder = CLIP(freeze_lm=True)
    denoiser = LightT2M(
        motion_dim=SOKE_TOTAL_DIM,
        max_motion_len=301,
        text_dim=512,
        pos_emb="cos",
        stage_dim="256*4",
        num_groups=16,
        patch_size=8,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        prediction_type="sample",
        clip_sample=False,
    )
    sample_scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        solver_order=2,
        prediction_type="sample",
    )
    
    ema_config = SimpleNamespace(use_ema=False)
    
    model = LightMotionGeneration(
        text_encoder=text_encoder,
        denoiser=denoiser,
        noise_scheduler=noise_scheduler,
        sample_scheduler=sample_scheduler,
        text_replace_prob=0.1,
        guidance_scale=args.guidance_scale,
        dataset_name='sign',
        evaluator=None,
        optimizer=None,
        ema=ema_config,
        lr_scheduler=None,
        step_num=args.num_inference_steps,
    )
    
    # Load checkpoint
    print(f"  Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    if 'epoch' in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if 'global_step' in ckpt:
        print(f"  Global step: {ckpt['global_step']}")
    
    model.eval()
    model.to(device)
    
    # =========================
    # Print Model Info
    # =========================
    print(f"\n[Model Info]")
    print(f"  Model type: Light-T2M")
    print(f"  Feature dim: {SOKE_TOTAL_DIM}")
    print(f"  Diffusion steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    
    # Get feats2joints converter
    feats2joints = datamodule.feats2joints
    print(f"  feats2joints: Available (from datamodule)")
    
    # =========================
    # Process Each Split
    # =========================
    print(f"\n[3/3] Processing samples...")
    
    all_metrics = []
    global_sample_idx = 0
    
    for split in splits_to_process:
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}")
        
        # Get dataloader
        try:
            if split == 'train':
                datamodule.setup(stage='fit')
                dataloader = datamodule.train_dataloader()
            elif split == 'val':
                datamodule.setup(stage='fit')
                dataloader = datamodule.val_dataloader()
            else:
                datamodule.setup(stage='test')
                dataloader = datamodule.test_dataloader()
        except Exception as e:
            print(f"  Warning: Could not load {split} dataloader: {e}")
            continue
        
        # Create output subdirectory
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Track collected samples
        collected = {ds: 0 for ds in datasets_to_process}
        target_per_dataset = args.num_samples
        
        def all_collected():
            return all(collected[ds] >= target_per_dataset for ds in datasets_to_process)
        
        for batch in tqdm(dataloader, desc=f"{split}"):
            if all_collected():
                break
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            feats_ref = batch['motion']
            texts = batch['text']
            lengths = batch['motion_len']
            names = batch.get('name', [f'sample_{i}' for i in range(len(lengths))])
            srcs = batch.get('src', ['how2sign'] * len(lengths))
            B = feats_ref.shape[0]
            
            with torch.no_grad():
                for i in range(B):
                    if all_collected():
                        break
                    
                    length = int(lengths[i].item() if isinstance(lengths[i], torch.Tensor) else lengths[i])
                    if length == 0:
                        continue
                    
                    name = names[i] if names else f'sample_{global_sample_idx}'
                    src = srcs[i] if srcs else 'unknown'
                    text = texts[i]
                    
                    src_key = normalize_source_name(src)
                    
                    if src_key not in datasets_to_process:
                        continue
                    if collected[src_key] >= target_per_dataset:
                        continue
                    
                    # Forward through LM (Text → Motion via Diffusion)
                    feats_pred = lm_forward(
                        model, text, length, device,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                    )
                    
                    # Convert to joints
                    has_joints = False
                    gt_joints_np = None
                    recon_joints_np = None
                    
                    try:
                        gt_input = feats_ref[i:i+1, :length]
                        recon_input = feats_pred[:, :length]
                        
                        gt_joints = feats2joints(gt_input)
                        recon_joints = feats2joints(recon_input)
                        
                        # Apply POSE_SCALE
                        gt_joints = gt_joints * POSE_SCALE
                        recon_joints = recon_joints * POSE_SCALE
                        
                        gt_joints_np = gt_joints[0].cpu().numpy()
                        recon_joints_np = recon_joints[0].cpu().numpy()
                        
                        has_joints = True
                    except Exception as e:
                        if global_sample_idx == 0:
                            print(f"    Note: feats2joints failed ({e})")
                        has_joints = False
                    
                    # Get feature arrays (denormalized)
                    gt_feats_np = datamodule.denormalize(feats_ref[i, :length]).cpu().numpy()
                    recon_feats_np = datamodule.denormalize(feats_pred[0, :length]).cpu().numpy()
                    
                    # Check for NaN
                    has_nan = np.isnan(recon_feats_np).any()
                    
                    # Compute metrics
                    metrics = compute_metrics(gt_feats_np, recon_feats_np, gt_joints_np, recon_joints_np)
                    metrics['name'] = name
                    metrics['text'] = text[:200]
                    metrics['src'] = src_key
                    metrics['split'] = split
                    metrics['length'] = int(length)
                    metrics['has_nan'] = has_nan
                    all_metrics.append(metrics)
                    
                    # Print sample info
                    print(f"\n[{split}/{src_key}] {name}")
                    print(f"  Text: {text[:60]}...")
                    print(f"  Length: {length} frames")
                    
                    if has_nan:
                        print(f"  ⚠ NaN detected in prediction (model may need more training)")
                    else:
                        print(f"  Feature RMSE: {metrics['feat_rmse']:.6f}")
                        if 'feat_rmse_body' in metrics:
                            print(f"  Feature RMSE - Body: {metrics['feat_rmse_body']:.6f}, "
                                  f"LHand: {metrics['feat_rmse_lhand']:.6f}, "
                                  f"RHand: {metrics['feat_rmse_rhand']:.6f}")
                        if 'feat_rmse_jaw' in metrics:
                            print(f"  Feature RMSE - Jaw: {metrics['feat_rmse_jaw']:.6f}, "
                                  f"Expr: {metrics['feat_rmse_expr']:.6f}")
                        if has_joints and 'mpjpe' in metrics and not np.isnan(metrics.get('mpjpe', float('nan'))):
                            print(f"  MPJPE: {metrics['mpjpe']:.4f}")
                    
                    # Save video
                    if has_joints and not args.no_video and not has_nan:
                        ds_output_dir = os.path.join(split_output_dir, src_key)
                        os.makedirs(ds_output_dir, exist_ok=True)
                        
                        safe_name = name.replace('/', '_').replace('\\', '_')[:40]
                        
                        if args.three_view:
                            video_path = os.path.join(ds_output_dir, 
                                                      f'{collected[src_key]:03d}_{safe_name}_3view.mp4')
                            save_three_view_video(gt_joints_np, recon_joints_np, video_path,
                                                 f'{split}/{src_key}: {name}', args.fps)
                        else:
                            video_path = os.path.join(ds_output_dir, 
                                                      f'{collected[src_key]:03d}_{safe_name}.mp4')
                            save_comparison_video(gt_joints_np, recon_joints_np, video_path,
                                                 f'{split}/{src_key}: {name}', args.fps)
                        print(f"  Saved: {video_path}")
                    elif has_nan:
                        print(f"  Skipped video (NaN in prediction)")
                    
                    collected[src_key] += 1
                    global_sample_idx += 1
        
        # Print collection summary
        print(f"\n[{split}] Collection summary:")
        for ds, count in collected.items():
            print(f"  {ds}: {count}/{target_per_dataset}")
    
    # =========================
    # Print Summary
    # =========================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_metrics)}")
    print(f"Model: Light-T2M")
    print(f"Feature dim: {SOKE_TOTAL_DIM}")
    print(f"Diffusion steps: {args.num_inference_steps}, Guidance: {args.guidance_scale}")
    
    # Count valid samples
    valid_metrics = [m for m in all_metrics if not m.get('has_nan', False)]
    nan_count = len(all_metrics) - len(valid_metrics)
    
    print(f"Valid samples: {len(valid_metrics)}")
    if nan_count > 0:
        print(f"NaN samples: {nan_count} (model may need more training)")
    
    if valid_metrics:
        print(f"\n[Overall]")
        print(f"  Avg Feature RMSE: {np.mean([m['feat_rmse'] for m in valid_metrics]):.6f}")
        
        if 'feat_rmse_body' in valid_metrics[0]:
            print(f"  Avg Feature RMSE - Body: {np.mean([m['feat_rmse_body'] for m in valid_metrics]):.6f}")
            print(f"  Avg Feature RMSE - LHand: {np.mean([m['feat_rmse_lhand'] for m in valid_metrics]):.6f}")
            print(f"  Avg Feature RMSE - RHand: {np.mean([m['feat_rmse_rhand'] for m in valid_metrics]):.6f}")
        
        if 'feat_rmse_jaw' in valid_metrics[0]:
            print(f"  Avg Feature RMSE - Jaw: {np.mean([m['feat_rmse_jaw'] for m in valid_metrics]):.6f}")
            print(f"  Avg Feature RMSE - Expr: {np.mean([m['feat_rmse_expr'] for m in valid_metrics]):.6f}")
        
        mpjpe_metrics = [m for m in valid_metrics if 'mpjpe' in m and not np.isnan(m.get('mpjpe', float('nan')))]
        if mpjpe_metrics:
            print(f"  Avg MPJPE: {np.mean([m['mpjpe'] for m in mpjpe_metrics]):.4f}")
    
    # =========================
    # Save Metrics to JSON
    # =========================
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, 
                  default=lambda x: x.item() if hasattr(x, 'item') else x)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save config summary
    config_summary = {
        'checkpoint': args.checkpoint,
        'timestamp': timestamp,
        'num_samples_per_dataset_per_split': args.num_samples,
        'splits': splits_to_process,
        'datasets': datasets_to_process,
        'model_type': 'Light-T2M',
        'feature_dim': SOKE_TOTAL_DIM,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'total_samples': len(all_metrics),
        'valid_samples': len(valid_metrics),
        'nan_samples': nan_count,
    }
    
    config_path = os.path.join(output_dir, 'config_summary.json')
    with open(config_path, 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    print(f"Config summary saved to: {config_path}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()