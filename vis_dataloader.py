"""
vis_dataloader.py — Sign-t2m 데이터로더 검증 시각화

DataModule로 로드된 데이터를 denormalize → SMPL-X FK (or approximate) → skeleton 영상 저장
VAE 없이 GT 데이터만 시각화하여 데이터 파이프라인이 정상인지 확인

Usage:
    cd ~/Projects/research/Sign-t2m

    # 기본 (H2S + CSL + Phoenix, val split)
    python vis_dataloader.py

    # How2Sign만, train split, 10개 샘플
    python vis_dataloader.py --dataset how2sign --split train --num_samples 10

    # 6D (240-dim) 데이터 확인
    python vis_dataloader.py --nfeats 240

    # viewport, fps 조정
    python vis_dataloader.py --viewport 0.8 --fps 15
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ── project root ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Constants — SMPL-X joint indices
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_VALID = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND


# =============================================================================
# Skeleton Visualization (from vis_sign_recon.py)
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
                hand_connections.extend([(20, base), (base, base + 1), (base + 1, base + 2)])
        for finger in range(5):
            base = 40 + finger * 3
            if base + 2 < num_joints:
                hand_connections.extend([(21, base), (base, base + 1), (base + 1, base + 2)])
    return [(i, j) for i, j in upper_body + hand_connections if i < num_joints and j < num_joints]


def normalize_to_root(joints, root_idx=9):
    if len(joints.shape) == 3:
        root = joints[:, root_idx:root_idx + 1, :]
    else:
        root = joints[root_idx:root_idx + 1, :]
    return joints - root


def save_skeleton_video(joints, save_path, title='', fps=25, viewport=0.5):
    """Single skeleton video (GT only)"""
    T, J, _ = joints.shape
    root_idx = 9 if J > 21 else 0
    data = normalize_to_root(joints.copy(), root_idx)

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        valid_idx = [i for i in SMPLX_VALID if i < J]
        all_data = data[:, valid_idx]
        all_x, all_y = all_data[:, :, 0].flatten(), all_data[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid, y_mid = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range / 2, x_mid + max_range / 2)
        y_lim = (y_mid - max_range / 2, y_mid + max_range / 2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(title, fontsize=10)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.axis('off')

    connections = get_connections(J)
    colors = {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}
    ub_idx = [i for i in SMPLX_UPPER_BODY if i < J]

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

    bs = ax.scatter([], [], c=colors['body'], s=10, zorder=5)
    ls = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
    rs = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T - 1}')
        fd = data[f]
        x, y = fd[:, 0], fd[:, 1]
        for (line, i, j) in lines:
            line.set_data([x[i], x[j]], [y[i], y[j]])
        bs.set_offsets(np.c_[x[ub_idx], y[ub_idx]])
        if J > 25:
            ls.set_offsets(np.c_[x[25:40], y[25:40]])
        if J > 40:
            rs.set_offsets(np.c_[x[40:55], y[40:55]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# Feature → Joint conversion
# =============================================================================

def feats_to_joints_approximate(features_np):
    """
    120D axis-angle → approximate 55-joint positions (no SMPL-X needed)

    Interprets raw axis-angle values as pseudo-positions for quick visualization.
    Not physically accurate, but sufficient to verify data loading correctness.

    Args:
        features_np: [T, 120] denormalized axis-angle
    Returns:
        joints: [T, 55, 3]
    """
    T, D = features_np.shape
    joints = np.zeros((T, 55, 3), dtype=np.float32)

    if D >= 120:
        upper_body = features_np[:, 0:30].reshape(T, 10, 3)
        lhand = features_np[:, 30:75].reshape(T, 15, 3)
        rhand = features_np[:, 75:120].reshape(T, 15, 3)
        joints[:, 12:22, :] = upper_body
        joints[:, 25:40, :] = lhand
        joints[:, 40:55, :] = rhand
    return joints


def feats_to_joints_smplx(features_np, mean, std, device='cuda:0'):
    """
    120D → SMPL-X FK → 55 joints (accurate)

    Falls back to approximate if SMPL-X unavailable.
    """
    try:
        from src.utils.feats2joints import feats2joints_smplx
        features_t = torch.from_numpy(features_np).float().unsqueeze(0)  # [1, T, 120]
        # feats2joints_smplx expects *normalized* input and denormalizes internally
        mean_t = mean.float()
        std_t = std.float()
        # Re-normalize (our input is already denormalized, so we need to undo)
        features_norm = (features_t - mean_t) / (std_t + 1e-10)
        _, joints = feats2joints_smplx(features_norm.to(device), mean_t.to(device), std_t.to(device))
        return joints.squeeze(0).cpu().numpy()  # [T, J, 3]
    except Exception as e:
        print(f"  SMPL-X FK failed ({e}), using approximate joints")
        return feats_to_joints_approximate(features_np)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sign-t2m DataLoader Visualization')
    # Data paths
    parser.add_argument('--data_root', default='/home/user/Projects/research/SOKE/data/How2Sign')
    parser.add_argument('--csl_root', default='/home/user/Projects/research/SOKE/data/CSL-Daily')
    parser.add_argument('--phoenix_root', default='/home/user/Projects/research/SOKE/data/Phoenix_2014T')
    parser.add_argument('--mean_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily/mean_120.pt')
    parser.add_argument('--std_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily/std_120.pt')
    parser.add_argument('--csl_mean_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_mean_120.pt')
    parser.add_argument('--csl_std_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_std_120.pt')
    # Dataset
    parser.add_argument('--dataset', default='how2sign_csl_phoenix',
                        choices=['how2sign', 'csl', 'how2sign_csl', 'how2sign_csl_phoenix'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--nfeats', type=int, default=120)
    parser.add_argument('--max_motion_length', type=int, default=300)
    parser.add_argument('--min_motion_length', type=int, default=40)
    # Visualization
    parser.add_argument('--num_samples', type=int, default=3, help='samples per dataset source')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5, help='0=auto')
    parser.add_argument('--output', default='vis_dataloader_output')
    # FK mode
    parser.add_argument('--use_smplx', action='store_true', help='use SMPL-X FK (needs deps/smpl_models)')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'dataloader_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    # =========================================================================
    # 1. Load mean/std
    # =========================================================================
    print("=" * 60)
    print("Sign-t2m DataLoader Visualization")
    print("=" * 60)

    mean = torch.load(args.mean_path, map_location='cpu').float()
    std = torch.load(args.std_path, map_location='cpu').float()
    mean = mean[:args.nfeats]
    std = std[:args.nfeats]

    csl_mean = mean  # fallback
    csl_std = std
    if os.path.exists(args.csl_mean_path):
        csl_mean = torch.load(args.csl_mean_path, map_location='cpu').float()[:args.nfeats]
    if os.path.exists(args.csl_std_path):
        csl_std = torch.load(args.csl_std_path, map_location='cpu').float()[:args.nfeats]

    print(f"  mean shape: {mean.shape}, range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std shape:  {std.shape},  range: [{std.min():.4f}, {std.max():.4f}]")
    print(f"  csl_mean:   [{csl_mean.min():.4f}, {csl_mean.max():.4f}]")
    print(f"  csl_std:    [{csl_std.min():.4f}, {csl_std.max():.4f}]")

    # =========================================================================
    # 2. Build Dataset
    # =========================================================================
    print(f"\n[1/3] Loading {args.dataset} / {args.split} ...")

    from src.data.signlang.dataset_sign import SignText2MotionDataset
    from src.data.signlang.collate import sign_collate

    dataset = SignText2MotionDataset(
        data_root=args.data_root,
        csl_root=args.csl_root,
        phoenix_root=args.phoenix_root,
        split=args.split,
        mean=mean,
        std=std,
        csl_mean=csl_mean,
        csl_std=csl_std,
        nfeats=args.nfeats,
        dataset_name=args.dataset,
        max_motion_length=args.max_motion_length,
        min_motion_length=args.min_motion_length,
    )

    print(f"  Total samples: {len(dataset)}")

    # =========================================================================
    # 3. Data Statistics (quick scan)
    # =========================================================================
    print(f"\n[2/3] Data statistics (scanning first 200 samples)...")

    scan_n = min(200, len(dataset))
    lengths = []
    feat_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    src_counts = {}

    for i in range(scan_n):
        item = dataset[i]
        if item is None:
            continue
        motion = item['motion'].numpy()  # [T, 120] normalized
        src = item.get('src', 'unknown')
        lengths.append(item['motion_len'])
        feat_stats['min'].append(motion.min())
        feat_stats['max'].append(motion.max())
        feat_stats['mean'].append(motion.mean())
        feat_stats['std'].append(motion.std())
        src_counts[src] = src_counts.get(src, 0) + 1

    lengths = np.array(lengths)
    print(f"  Scanned: {scan_n} samples")
    print(f"  Sources: {src_counts}")
    print(f"  Lengths: min={lengths.min()}, max={lengths.max()}, "
          f"mean={lengths.mean():.1f}, median={np.median(lengths):.0f}")
    print(f"  Feature (normalized):")
    print(f"    min:  {np.mean(feat_stats['min']):.4f}")
    print(f"    max:  {np.mean(feat_stats['max']):.4f}")
    print(f"    mean: {np.mean(feat_stats['mean']):.4f}")
    print(f"    std:  {np.mean(feat_stats['std']):.4f}")

    # =========================================================================
    # 4. Batch test (DataLoader)
    # =========================================================================
    print(f"\n  DataLoader batch test...")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        collate_fn=sign_collate, num_workers=0)
    batch = next(iter(loader))
    print(f"  Batch keys:    {list(batch.keys())}")
    print(f"  motion shape:  {batch['motion'].shape}")
    print(f"  motion_len:    {batch['motion_len'].tolist()}")
    print(f"  texts[0]:      {batch['text'][0][:80]}...")
    print(f"  names[0]:      {batch['name'][0]}")

    # =========================================================================
    # 5. Visualize samples
    # =========================================================================
    print(f"\n[3/3] Generating skeleton videos...")

    DS_LABELS = {'how2sign': 'H2S', 'csl': 'CSL', 'phoenix': 'Phoenix'}
    total_count = 0

    # Group by source
    src_indices = {}
    for i, item in enumerate(dataset.all_data):
        s = item.get('src', 'how2sign')
        src_indices.setdefault(s, []).append(i)

    for src_key, indices in src_indices.items():
        if not indices:
            continue

        ds_label = DS_LABELS.get(src_key, src_key)
        n = min(args.num_samples, len(indices))
        # Evenly spaced samples
        sel = [indices[int(i)] for i in np.linspace(0, len(indices) - 1, n)]

        out_dir = os.path.join(output_root, ds_label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  [{ds_label}] {len(indices)} total, visualizing {n}")

        for idx_in_sel, dataset_idx in enumerate(sel):
            item = dataset[dataset_idx]
            if item is None:
                print(f"    [{idx_in_sel + 1}/{n}] load failed, skip.")
                continue

            motion_norm = item['motion'].numpy()    # [T, nfeats] normalized
            text = item['text']
            name = item['name']
            src = item.get('src', src_key)
            T_len = item['motion_len']

            # Denormalize (dataset-specific mean/std)
            if src == 'csl':
                m_np, s_np = csl_mean.numpy(), csl_std.numpy()
            else:
                m_np, s_np = mean.numpy(), std.numpy()

            motion_raw = motion_norm * (s_np + 1e-10) + m_np  # [T, 120]

            # Convert to joints
            if args.use_smplx:
                m_t = csl_mean if src == 'csl' else mean
                s_t = csl_std if src == 'csl' else std
                joints = feats_to_joints_smplx(motion_raw, m_t, s_t, args.device)
            else:
                joints = feats_to_joints_approximate(motion_raw)

            J = joints.shape[1]

            # NaN check
            if np.isnan(joints).any():
                print(f"    [{idx_in_sel + 1}/{n}] {name} — NaN in joints, skip.")
                continue

            # Stats
            raw_rmse = np.sqrt(np.mean(motion_raw ** 2))
            print(f"    [{idx_in_sel + 1}/{n}] {name} (T={T_len}, J={J}, src={src})")
            print(f"      text: {text[:70]}...")
            print(f"      raw RMSE: {raw_rmse:.4f}, "
                  f"norm range: [{motion_norm.min():.3f}, {motion_norm.max():.3f}]")

            # Save video
            safe_name = str(name)[:40].replace('/', '_').replace('\\', '_')
            video_path = os.path.join(out_dir, f'{idx_in_sel:03d}_{safe_name}.mp4')
            title = f'{name}\n[{ds_label}] T={T_len}  {text[:50]}...'

            save_skeleton_video(joints, video_path, title, args.fps, args.viewport)
            total_count += 1

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Done. {total_count} videos saved to {output_root}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Split:    {args.split}")
    print(f"  nfeats:   {args.nfeats}")
    print(f"  FK mode:  {'SMPL-X' if args.use_smplx else 'Approximate (axis-angle as position)'}")
    if not args.use_smplx:
        print(f"  Note: approximate mode는 axis-angle을 position으로 해석합니다.")
        print(f"        정확한 skeleton은 --use_smplx 플래그를 사용하세요.")
    print("=" * 60)


if __name__ == '__main__':
    main()
