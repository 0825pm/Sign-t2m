"""
vis_vae.py — Sign-t2m VAE Reconstruction 시각화

학습된 VAE 체크포인트에서 motion → encode → decode → GT vs Reconstructed 비교 영상

Usage:
    cd ~/Projects/research/sign-t2m

    # Val set에서 10개 샘플 reconstruction 비교
    python vis_vae.py \
        --ckpt logs/sign-vae/runs/.../checkpoints/last.ckpt \
        --num_samples 10

    # 특정 데이터셋만
    python vis_vae.py \
        --ckpt logs/.../last.ckpt \
        --dataset how2sign --num_samples 5

    # 자동 viewport (skeleton 크기에 맞춤)
    python vis_vae.py \
        --ckpt logs/.../last.ckpt \
        --viewport 0
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Constants
# =============================================================================
SMPLX_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_VALID = SMPLX_UPPER_BODY + SMPLX_LHAND + SMPLX_RHAND

# 44 joints → SMPLX 55 index mapping
JOINT44_TO_SMPLX55 = (
    [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    + list(range(25, 40))
    + list(range(40, 55))
)


# =============================================================================
# Skeleton Visualization (vis_generation.py 동일 구조)
# =============================================================================

def get_connections():
    upper_body = [
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 13), (13, 16), (16, 18), (18, 20),
        (9, 14), (14, 17), (17, 19), (19, 21),
    ]
    hand_connections = []
    for finger in range(5):
        base = 25 + finger * 3
        hand_connections.extend([(20, base), (base, base + 1), (base + 1, base + 2)])
    for finger in range(5):
        base = 40 + finger * 3
        hand_connections.extend([(21, base), (base, base + 1), (base + 1, base + 2)])
    return upper_body + hand_connections


def normalize_to_root(joints, root_idx=9, flip_y=True):
    if len(joints.shape) == 3:
        root = joints[:, root_idx:root_idx + 1, :]
    else:
        root = joints[root_idx:root_idx + 1, :]
    out = joints - root
    if flip_y:
        out[..., 1] *= -1  # Y축 반전 (528D 등 Z-up 좌표계용)
    return out


def _setup_ax(ax, label, color, x_lim, y_lim):
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.axis('off')


def _build_elements(ax, J, connections, colors):
    ub_idx = [i for i in SMPLX_UPPER_BODY if i < J]
    lines = []
    for (i, j) in connections:
        if i >= J or j >= J:
            continue
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
    return lines, bs, ls, rs, ub_idx


def save_comparison_video(left_joints, right_joints, save_path,
                          title='', fps=25, viewport=0.5,
                          left_label='GT', right_label='Reconstructed',
                          flip_y=True):
    T = min(left_joints.shape[0], right_joints.shape[0])
    J = min(left_joints.shape[1], right_joints.shape[1])

    root_idx = 9 if J > 21 else 0
    left = normalize_to_root(left_joints[:T, :J].copy(), root_idx, flip_y=flip_y)
    right = normalize_to_root(right_joints[:T, :J].copy(), root_idx, flip_y=flip_y)

    valid_idx = [i for i in SMPLX_VALID if i < J]

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        all_data = np.concatenate([left[:, valid_idx], right[:, valid_idx]], axis=0)
        all_x, all_y = all_data[:, :, 0].flatten(), all_data[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range / 2, x_mid + max_range / 2)
        y_lim = (y_mid - max_range / 2, y_mid + max_range / 2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=10)

    _setup_ax(ax_l, left_label, 'blue', x_lim, y_lim)
    _setup_ax(ax_r, right_label, 'red', x_lim, y_lim)

    connections = get_connections()
    colors_l = {'body': '#000000', 'lhand': '#E91E63', 'rhand': '#4CAF50'}
    colors_r = {'body': '#000000', 'lhand': '#E91E63', 'rhand': '#4CAF50'}

    el_l = _build_elements(ax_l, J, connections, colors_l)
    el_r = _build_elements(ax_r, J, connections, colors_r)

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T - 1}')
        for (lines, bs, ls, rs, ub_idx), data in [(el_l, left), (el_r, right)]:
            fd = data[f]
            x, y = fd[:, 0], fd[:, 1]
            for (line, i, j) in lines:
                line.set_data([x[i], x[j]], [y[i], y[j]])
            bs.set_offsets(np.c_[x[ub_idx], y[ub_idx]])
            if J > 25:
                ls.set_offsets(np.c_[x[25:min(40, J)], y[25:min(40, J)]])
            if J > 40:
                rs.set_offsets(np.c_[x[40:min(55, J)], y[40:min(55, J)]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# 528D → Joint positions
# =============================================================================

def feats528_to_joints55(features_np, mean_np=None, std_np=None):
    """
    528D features → 55-joint positions (SMPLX format).
    528D = positions(132) + velocities(132) + 6D_rot(264)
    positions = 44 joints × 3
    """
    T = features_np.shape[0]

    if mean_np is not None and std_np is not None:
        features_np = features_np * (std_np + 1e-10) + mean_np

    pos_44 = features_np[:, :132].reshape(T, 44, 3)

    joints = np.zeros((T, 55, 3), dtype=np.float32)
    for local_idx, smplx_idx in enumerate(JOINT44_TO_SMPLX55):
        joints[:, smplx_idx, :] = pos_44[:, local_idx, :]
    return joints


def feats120_to_joints55(features_np):
    """120D axis-angle → approximate 55-joint positions"""
    T, D = features_np.shape
    joints = np.zeros((T, 55, 3), dtype=np.float32)
    if D >= 120:
        joints[:, 12:22, :] = features_np[:, 0:30].reshape(T, 10, 3)
        joints[:, 25:40, :] = features_np[:, 30:75].reshape(T, 15, 3)
        joints[:, 40:55, :] = features_np[:, 75:120].reshape(T, 15, 3)
    return joints


# =============================================================================
# Model Loading
# =============================================================================

def load_vae_model(ckpt_path, device):
    """Load SignVAE from Lightning checkpoint."""
    from src.models.sign_vae import SignVAE
    from src.models.nets.motion_vae import MldVae
    from functools import partial

    print(f"  Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})

    # Build VAE from checkpoint hparams + state_dict (구조 자동 감지)
    state = ckpt['state_dict']

    # Infer nfeats from skel_embedding
    skel_emb_weight = state.get('vae.skel_embedding.weight')
    if skel_emb_weight is not None:
        nfeats = skel_emb_weight.shape[1]
    else:
        nfeats = 528

    # Infer num_layers from encoder block keys
    enc_keys = [k for k in state if k.startswith('vae.encoder.input_blocks.')]
    num_input_blocks = len(set(k.split('.')[3] for k in enc_keys)) if enc_keys else 4
    num_layers = num_input_blocks * 2 + 1  # input + middle + output

    # Infer ff_size from first FFN layer
    ffn_key = 'vae.encoder.input_blocks.0.linear1.weight'
    ff_size = state[ffn_key].shape[0] if ffn_key in state else 1024

    # Infer latent_dim from dist token
    dist_key = 'vae.global_motion_token'
    if dist_key in state:
        latent_dim = [state[dist_key].shape[0] // 2, state[dist_key].shape[1]]
    else:
        latent_dim = [1, 256]

    print(f"  Detected: nfeats={nfeats}, layers={num_layers}, ff={ff_size}, latent={latent_dim}")

    vae = MldVae(
        nfeats=nfeats,
        latent_dim=latent_dim,
        ff_size=ff_size,
        num_layers=num_layers,
        num_heads=4,
        dropout=0.0,  # eval mode, dropout 무관
        activation='gelu',
    )

    model = SignVAE(
        vae=vae,
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
        **{k: v for k, v in hparams.items()
           if k not in ('vae', 'optimizer', 'lr_scheduler')},
    )

    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if missing:
        print(f"  ⚠️  Missing: {missing[:5]}")
    if unexpected:
        print(f"  ⚠️  Unexpected: {unexpected[:5]}")

    model.eval().to(device)

    n_params = sum(p.numel() for p in model.vae.parameters())
    epoch = ckpt.get('epoch', '?')
    step = ckpt.get('global_step', '?')
    print(f"  Epoch: {epoch}, Step: {step}, VAE: {n_params / 1e6:.2f}M")

    return model, nfeats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sign-t2m VAE Reconstruction Visualization')
    parser.add_argument('--ckpt', required=True)
    # Data paths
    parser.add_argument('--data_root',
        default='/home/user/Projects/research/SOKE/data/How2Sign')
    parser.add_argument('--csl_root',
        default='/home/user/Projects/research/SOKE/data/CSL-Daily')
    parser.add_argument('--phoenix_root',
        default='/home/user/Projects/research/SOKE/data/Phoenix_2014T')
    parser.add_argument('--npy_root', default=None)
    parser.add_argument('--csl_npy_root', default=None)
    parser.add_argument('--phoenix_npy_root', default=None)
    parser.add_argument('--mean_path', default=None)
    parser.add_argument('--std_path', default=None)
    parser.add_argument('--csl_mean_path', default=None)
    parser.add_argument('--csl_std_path', default=None)
    # Dataset
    parser.add_argument('--dataset', default='how2sign_csl_phoenix')
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=10)
    # Vis
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5,
                        help='fixed viewport (0=auto)')
    parser.add_argument('--output', default='vis_vae_output')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'vae_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("Sign-t2m VAE Reconstruction Visualization")
    print("=" * 60)

    # =========================================================================
    # 1. Load Model
    # =========================================================================
    print("\n[1/3] Loading VAE...")
    model, nfeats = load_vae_model(args.ckpt, device)

    # =========================================================================
    # 2. Load normalization stats (auto-detect paths from nfeats)
    # =========================================================================
    print("\n[2/3] Loading data...")
    D = nfeats
    BASE = '/home/user/Projects/research/SOKE/data'

    if D >= 528:
        mean_path = args.mean_path or f'{BASE}/How2Sign_528d/mean_528.pt'
        std_path = args.std_path or f'{BASE}/How2Sign_528d/std_528.pt'
        csl_mean_path = args.csl_mean_path or f'{BASE}/CSL-Daily_528d/mean_528.pt'
        csl_std_path = args.csl_std_path or f'{BASE}/CSL-Daily_528d/std_528.pt'
        npy_root = args.npy_root or f'{BASE}/How2Sign_528d'
        csl_npy_root = args.csl_npy_root or f'{BASE}/CSL-Daily_528d'
        phoenix_npy_root = args.phoenix_npy_root or f'{BASE}/Phoenix_528d'
    else:
        mean_path = args.mean_path or f'{BASE}/CSL-Daily/mean_120.pt'
        std_path = args.std_path or f'{BASE}/CSL-Daily/std_120.pt'
        csl_mean_path = args.csl_mean_path or f'{BASE}/CSL-Daily/csl_mean_120.pt'
        csl_std_path = args.csl_std_path or f'{BASE}/CSL-Daily/csl_std_120.pt'
        npy_root = args.npy_root
        csl_npy_root = args.csl_npy_root
        phoenix_npy_root = args.phoenix_npy_root

    print(f"  nfeats={D}, mean: {mean_path}")

    mean = torch.load(mean_path, map_location='cpu').float()[:D]
    std = torch.load(std_path, map_location='cpu').float()[:D]

    csl_mean, csl_std = mean, std
    if csl_mean_path and os.path.exists(csl_mean_path):
        csl_mean = torch.load(csl_mean_path, map_location='cpu').float()[:D]
    if csl_std_path and os.path.exists(csl_std_path):
        csl_std = torch.load(csl_std_path, map_location='cpu').float()[:D]

    # =========================================================================
    # 3. Load dataset (same class as training)
    # =========================================================================
    from src.data.signlang.dataset_sign import SignMotionDataset

    dataset = SignMotionDataset(
        data_root=args.data_root,
        split=args.split,
        mean=mean,
        std=std,
        nfeats=D,
        dataset_name=args.dataset,
        max_motion_length=400,
        min_motion_length=20,
        csl_root=args.csl_root,
        phoenix_root=args.phoenix_root,
        npy_root=npy_root,
        csl_npy_root=csl_npy_root,
        phoenix_npy_root=phoenix_npy_root,
        csl_mean=csl_mean,
        csl_std=csl_std,
    )

    n = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    print(f"  Dataset: {len(dataset)} samples, visualizing {n}")

    # =========================================================================
    # 4. Reconstruct & Visualize
    # =========================================================================
    print(f"\n[3/3] Generating reconstructions...\n")

    rmse_all_list, body_list, hand_list = [], [], []

    # feats→joints 함수 선택
    if D >= 528:
        def to_joints(feats_norm, m_np, s_np):
            return feats528_to_joints55(feats_norm, m_np, s_np)
    else:
        # 120D axis-angle → SMPLX FK
        from src.utils.feats2joints import feats2joints_smplx

        def to_joints(feats_norm, m_np, s_np):
            t = torch.from_numpy(feats_norm).float().unsqueeze(0).to(device)
            m_t = torch.from_numpy(m_np).float().to(device)
            s_t = torch.from_numpy(s_np).float().to(device)
            try:
                _, joints = feats2joints_smplx(t, m_t, s_t)
                return joints.squeeze(0).cpu().numpy()[:, :55, :]
            except Exception as e:
                print(f"    SMPLX FK failed ({e}), using approximate")
                raw = feats_norm * (s_np + 1e-10) + m_np
                return feats120_to_joints55(raw)

    for idx_i, ds_idx in enumerate(indices):
        item = dataset[ds_idx]
        if item is None:
            continue

        gt_norm = item['motion']               # [T, D] tensor
        text = item.get('text', '')
        name = item.get('name', f'sample_{ds_idx}')
        src = item.get('src', 'how2sign')
        T_len = int(item['motion_len'])

        # Per-dataset mean/std for denormalization
        if src == 'csl':
            m_np = csl_mean.numpy()[:D]
            s_np = csl_std.numpy()[:D]
        else:
            m_np = mean.numpy()
            s_np = std.numpy()

        # VAE forward
        with torch.no_grad():
            motion_in = gt_norm.unsqueeze(0).float().to(device)  # [1, T, D]
            feats_rst, z, dist = model.vae(motion_in, [T_len])
            recon_norm = feats_rst[0].cpu().numpy()              # [T, D]

        gt_np = gt_norm.numpy()

        # ---- Metrics (normalized space) ----
        T = min(T_len, recon_norm.shape[0], gt_np.shape[0])
        gt_crop = gt_np[:T]
        recon_crop = recon_norm[:T]
        diff = gt_crop - recon_crop

        rmse = np.sqrt(np.mean(diff ** 2))

        if D >= 528:
            body_rmse = np.sqrt(np.mean(diff[:, :42] ** 2))     # 14 body × 3
            hand_rmse = np.sqrt(np.mean(diff[:, 42:132] ** 2))  # 30 hand × 3
        else:
            body_rmse = np.sqrt(np.mean(diff[:, :30] ** 2))
            hand_rmse = np.sqrt(np.mean(diff[:, 30:120] ** 2))

        rmse_all_list.append(rmse)
        body_list.append(body_rmse)
        hand_list.append(hand_rmse)

        # ---- Joints for visualization ----
        gt_joints = to_joints(gt_crop, m_np, s_np)
        recon_joints = to_joints(recon_crop, m_np, s_np)

        z_np = z.cpu().numpy()

        safe_name = str(name)[:30].replace('/', '_').replace(' ', '_')
        print(f"  [{idx_i + 1}/{n}] {name} (T={T_len}, src={src})")
        print(f"    RMSE: total={rmse:.4f}  body={body_rmse:.4f}  hand={hand_rmse:.4f}")
        print(f"    z: mean={z_np.mean():.3f}, std={z_np.std():.3f}")
        if text:
            print(f"    text: \"{text[:60]}\"")

        path = os.path.join(output_root, f'{idx_i:03d}_{safe_name}.mp4')
        title = (f'{name} [{src}] T={T_len}\n'
                 f'RMSE={rmse:.4f} (body={body_rmse:.4f}, hand={hand_rmse:.4f})')
        save_comparison_video(
            gt_joints, recon_joints, path, title,
            args.fps, args.viewport,
            flip_y=(D >= 528),  # 528D=Z-up→flip, 120D=SMPLX Y-up→no flip
        )
        print(f"    → {path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Summary ({n} samples)")
    print(f"  RMSE:      {np.mean(rmse_all_list):.4f} ± {np.std(rmse_all_list):.4f}")
    print(f"  Body RMSE: {np.mean(body_list):.4f} ± {np.std(body_list):.4f}")
    print(f"  Hand RMSE: {np.mean(hand_list):.4f} ± {np.std(hand_list):.4f}")
    print(f"\nVideos: {output_root}")
    print("=" * 60)

    # Save summary
    with open(os.path.join(output_root, 'summary.txt'), 'w') as f:
        f.write(f"VAE Reconstruction Summary\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Dataset: {args.dataset} / {args.split}\n")
        f.write(f"Samples: {n}\n\n")
        f.write(f"RMSE:      {np.mean(rmse_all_list):.4f} ± {np.std(rmse_all_list):.4f}\n")
        f.write(f"Body RMSE: {np.mean(body_list):.4f} ± {np.std(body_list):.4f}\n")
        f.write(f"Hand RMSE: {np.mean(hand_list):.4f} ± {np.std(hand_list):.4f}\n\n")
        for i, (r, b, h) in enumerate(zip(rmse_all_list, body_list, hand_list)):
            f.write(f"  [{i}] total={r:.4f}  body={b:.4f}  hand={h:.4f}\n")


if __name__ == '__main__':
    main()