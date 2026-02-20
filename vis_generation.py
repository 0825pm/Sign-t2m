"""
vis_generation.py — Sign-t2m Text→Motion 생성 시각화

학습된 체크포인트에서 text → motion 생성 후 skeleton 영상 저장
GT 비교 모드 / 자유 텍스트 생성 모드 지원

Usage:
    cd ~/Projects/research/Sign-t2m

    # GT 비교 (val set에서 N개 샘플 → GT vs Generated)
    python vis_generation.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --mode val --num_samples 5

    # 자유 텍스트 생성
    python vis_generation.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --mode text \
        --texts "a person waves hello" "pointing to the right" \
        --lengths 80 60

    # guidance scale / step 조정
    python vis_generation.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --mode val --guidance_scale 7.5 --step_num 20
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
from functools import partial

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


# =============================================================================
# Skeleton Visualization
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


def _setup_skeleton_ax(ax, label, color, x_lim, y_lim):
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.axis('off')


def _build_skeleton_elements(ax, J, connections, colors):
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
    return lines, bs, ls, rs, ub_idx


def save_comparison_video(left_joints, right_joints, save_path,
                          title='', fps=25, viewport=0.5,
                          left_label='GT', right_label='Generated'):
    """Side-by-side skeleton video"""
    T = min(left_joints.shape[0], right_joints.shape[0])
    J = min(left_joints.shape[1], right_joints.shape[1])

    root_idx = 9 if J > 21 else 0
    left = normalize_to_root(left_joints[:T, :J].copy(), root_idx)
    right = normalize_to_root(right_joints[:T, :J].copy(), root_idx)

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        valid_idx = [i for i in SMPLX_VALID if i < J]
        all_data = np.concatenate([left[:, valid_idx], right[:, valid_idx]], axis=0)
        all_x, all_y = all_data[:, :, 0].flatten(), all_data[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid, y_mid = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range/2, x_mid + max_range/2)
        y_lim = (y_mid - max_range/2, y_mid + max_range/2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=10)

    _setup_skeleton_ax(ax_l, left_label, 'blue', x_lim, y_lim)
    _setup_skeleton_ax(ax_r, right_label, 'red', x_lim, y_lim)

    connections = get_connections(J)
    colors = {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}

    elements = []
    for ax, data in [(ax_l, left), (ax_r, right)]:
        lines, bs, ls, rs, ub_idx = _build_skeleton_elements(ax, J, connections, colors)
        elements.append((lines, bs, ls, rs, ub_idx, data))

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        for (lines, bs, ls, rs, ub_idx, data) in elements:
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

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


def save_single_video(joints, save_path, title='', fps=25, viewport=0.5):
    """Single skeleton video (generation only, no GT)"""
    T, J, _ = joints.shape
    root_idx = 9 if J > 21 else 0
    data = normalize_to_root(joints.copy(), root_idx)

    if viewport > 0:
        x_lim = (-viewport, viewport)
        y_lim = (-viewport, viewport)
    else:
        valid_idx = [i for i in SMPLX_VALID if i < J]
        d = data[:, valid_idx]
        all_x, all_y = d[:, :, 0].flatten(), d[:, :, 1].flatten()
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), 0.1) * 1.2
        x_mid, y_mid = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
        x_lim = (x_mid - max_range/2, x_mid + max_range/2)
        y_lim = (y_mid - max_range/2, y_mid + max_range/2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(title, fontsize=10)
    _setup_skeleton_ax(ax, 'Generated', 'red', x_lim, y_lim)

    connections = get_connections(J)
    colors = {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}
    lines, bs, ls, rs, ub_idx = _build_skeleton_elements(ax, J, connections, colors)

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
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

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# Feature → Joint (approximate, no SMPL-X needed)
# =============================================================================

def feats_to_joints(features_np):
    """120D axis-angle → approximate 55-joint positions"""
    T, D = features_np.shape
    joints = np.zeros((T, 55, 3), dtype=np.float32)
    if D >= 120:
        joints[:, 12:22, :] = features_np[:, 0:30].reshape(T, 10, 3)
        joints[:, 25:40, :] = features_np[:, 30:75].reshape(T, 15, 3)
        joints[:, 40:55, :] = features_np[:, 75:120].reshape(T, 15, 3)
    return joints


def feats_to_joints_smplx(features_norm, mean, std, device='cuda:0'):
    """120D normalized → SMPL-X FK → joints (accurate)"""
    try:
        from src.utils.feats2joints import feats2joints_smplx
        t = torch.from_numpy(features_norm).float().unsqueeze(0).to(device)
        m = mean.float().to(device)
        s = std.float().to(device)
        _, joints = feats2joints_smplx(t, m, s)
        return joints.squeeze(0).cpu().numpy()
    except Exception as e:
        print(f"  SMPL-X FK failed ({e}), using approximate")
        raw = features_norm * (std.numpy() + 1e-10) + mean.numpy()
        return feats_to_joints(raw)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(ckpt_path, device, guidance_scale=None, step_num=None):
    """Load trained SignMotionGeneration from Lightning checkpoint"""
    from src.models.sign_t2m import SignMotionGeneration
    from src.models.nets.sign_denoiser import SignDenoiser
    from src.models.nets.text_encoder import CLIP
    from src.models.utils.utils import CosineWarmupScheduler
    from diffusers import DDPMScheduler, UniPCMultistepScheduler

    print(f"  Loading checkpoint: {ckpt_path}")

    # Load checkpoint to inspect hparams
    ckpt = torch.load(ckpt_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})

    # Build components
    text_encoder = CLIP(freeze_lm=True)

    denoiser = SignDenoiser(
        motion_dim=120,
        max_motion_len=401,
        text_dim=512,
        pos_emb="cos",
        stage_dim="256*4",
        num_groups=16,
        patch_size=8,
        ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
        rms_norm=False,
        fused_add_norm=True,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
        clip_sample=False, prediction_type="sample",
    )

    sample_scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", solver_order=2,
        prediction_type="sample",
    )

    gs = guidance_scale or hparams.get('guidance_scale', 4.0)
    sn = step_num or hparams.get('step_num', 10)

    model = SignMotionGeneration(
        text_encoder=text_encoder,
        denoiser=denoiser,
        noise_scheduler=noise_scheduler,
        sample_scheduler=sample_scheduler,
        text_replace_prob=0.0,
        guidance_scale=gs,
        dataset_name=hparams.get('dataset_name', 'how2sign'),
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
        lr_scheduler=None,
        step_num=sn,
        ema=hparams.get('ema', {"use_ema": False, "ema_decay": 0.99, "ema_start": 1000}),
    )

    # Load state_dict (text_encoder excluded from ckpt)
    state = ckpt['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)

    # missing은 text_encoder.* 이어야 정상
    non_te_missing = [k for k in missing if not k.startswith('text_encoder')]
    if non_te_missing:
        print(f"  ⚠️  Missing keys (non-text_encoder): {non_te_missing}")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {unexpected[:5]}")

    model.eval().to(device)

    n_params = sum(p.numel() for p in model.denoiser.parameters())
    epoch = ckpt.get('epoch', '?')
    step = ckpt.get('global_step', '?')
    print(f"  Epoch: {epoch}, Step: {step}, Denoiser: {n_params/1e6:.2f}M")
    print(f"  guidance_scale: {gs}, step_num: {sn}")

    return model


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sign-t2m Generation Visualization')
    # Checkpoint
    parser.add_argument('--ckpt', required=True, help='path to .ckpt')
    # Mode
    parser.add_argument('--mode', default='val', choices=['val', 'text'],
                        help='val: GT comparison / text: free generation')
    # Data (for val mode)
    parser.add_argument('--data_root', default='/home/user/Projects/research/SOKE/data/How2Sign')
    parser.add_argument('--csl_root', default='/home/user/Projects/research/SOKE/data/CSL-Daily')
    parser.add_argument('--phoenix_root', default='/home/user/Projects/research/SOKE/data/Phoenix_2014T')
    parser.add_argument('--mean_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily/mean_120.pt')
    parser.add_argument('--std_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily/std_120.pt')
    parser.add_argument('--csl_mean_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_mean_120.pt')
    parser.add_argument('--csl_std_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily/csl_std_120.pt')
    parser.add_argument('--dataset', default='how2sign')
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=5)
    # Text mode
    parser.add_argument('--texts', nargs='+', default=None,
                        help='text prompts for generation')
    parser.add_argument('--lengths', nargs='+', type=int, default=None,
                        help='motion lengths (frames) per text')
    # Generation
    parser.add_argument('--guidance_scale', type=float, default=None)
    parser.add_argument('--step_num', type=int, default=None)
    # Visualization
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0.5)
    parser.add_argument('--use_smplx', action='store_true')
    parser.add_argument('--output', default='vis_generation_output')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'gen_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("Sign-t2m Generation Visualization")
    print("=" * 60)

    # =========================================================================
    # 1. Load Model
    # =========================================================================
    print("\n[1/3] Loading model...")
    model = load_model(args.ckpt, device, args.guidance_scale, args.step_num)

    # =========================================================================
    # 2. Load mean/std for denormalization
    # =========================================================================
    mean = torch.load(args.mean_path, map_location='cpu').float()[:120]
    std = torch.load(args.std_path, map_location='cpu').float()[:120]
    mean_np, std_np = mean.numpy(), std.numpy()

    csl_mean, csl_std = mean, std
    if os.path.exists(args.csl_mean_path):
        csl_mean = torch.load(args.csl_mean_path, map_location='cpu').float()[:120]
    if os.path.exists(args.csl_std_path):
        csl_std = torch.load(args.csl_std_path, map_location='cpu').float()[:120]

    # =========================================================================
    # 3. Generate & Visualize
    # =========================================================================
    if args.mode == 'text':
        _run_text_mode(model, args, device, mean_np, std_np, mean, std, output_root)
    else:
        _run_val_mode(model, args, device, mean, std, mean_np, std_np,
                      csl_mean, csl_std, output_root)

    print(f"\n{'='*60}")
    print(f"Done. Videos saved to {output_root}")
    print("=" * 60)


def _run_text_mode(model, args, device, mean_np, std_np, mean, std, output_root):
    """Free text generation mode"""
    texts = args.texts or ["a person waves hello"]
    lengths = args.lengths or [100] * len(texts)
    assert len(texts) == len(lengths), "texts와 lengths 개수 맞춰야함"

    print(f"\n[2/3] Text mode: {len(texts)} prompts")

    for i, (text, length) in enumerate(zip(texts, lengths)):
        print(f"\n  [{i+1}/{len(texts)}] \"{text}\" (T={length})")

        with torch.no_grad():
            generated = model.generate([text], [length])  # [1, T, 120]

        gen_np = generated[0].cpu().numpy()  # [T, 120]
        gen_raw = gen_np * (std_np + 1e-10) + mean_np

        if args.use_smplx:
            joints = feats_to_joints_smplx(gen_np, mean, std, args.device)
        else:
            joints = feats_to_joints(gen_raw)

        print(f"    output range: [{gen_raw.min():.3f}, {gen_raw.max():.3f}]")

        safe_text = text[:40].replace(' ', '_').replace('/', '_')
        path = os.path.join(output_root, f'{i:03d}_{safe_text}.mp4')
        title = f'"{text}"\nT={length}'
        save_single_video(joints, path, title, args.fps, args.viewport)
        print(f"    saved: {path}")


def _run_val_mode(model, args, device, mean, std, mean_np, std_np,
                  csl_mean, csl_std, output_root):
    """GT comparison mode: load val data → generate from same text → side-by-side"""
    print(f"\n[2/3] Val mode: {args.dataset} / {args.split}")

    from src.data.signlang.dataset_sign import SignText2MotionDataset

    dataset = SignText2MotionDataset(
        data_root=args.data_root,
        csl_root=args.csl_root,
        phoenix_root=args.phoenix_root,
        split=args.split,
        mean=mean,
        std=std,
        csl_mean=csl_mean,
        csl_std=csl_std,
        nfeats=120,
        dataset_name=args.dataset,
        max_motion_length=400,
        min_motion_length=20,
    )

    n = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    print(f"  Dataset: {len(dataset)} samples, visualizing {n}")
    print(f"\n[3/3] Generating...")

    for idx_i, ds_idx in enumerate(indices):
        item = dataset[ds_idx]
        if item is None:
            continue

        gt_norm = item['motion'].numpy()           # [T, 120] normalized
        text = item['text']
        name = item['name']
        src = item.get('src', 'how2sign')
        T_len = item['motion_len']

        # Denormalize GT
        if src == 'csl':
            m_np = csl_mean.numpy()
            s_np = csl_std.numpy()
        else:
            m_np, s_np = mean_np, std_np
        gt_raw = gt_norm * (s_np + 1e-10) + m_np

        # Generate from same text + length
        with torch.no_grad():
            generated = model.generate([text], [T_len])  # [1, T, 120]
        gen_np = generated[0].cpu().numpy()
        gen_raw = gen_np * (std_np + 1e-10) + mean_np  # generation은 항상 기본 mean/std

        # Joints
        if args.use_smplx:
            m_t = csl_mean if src == 'csl' else mean
            s_t = csl_std if src == 'csl' else std
            gt_joints = feats_to_joints_smplx(gt_norm, m_t, s_t, args.device)
            gen_joints = feats_to_joints_smplx(gen_np, mean, std, args.device)
        else:
            gt_joints = feats_to_joints(gt_raw)
            gen_joints = feats_to_joints(gen_raw)

        # Metrics
        T = min(gt_raw.shape[0], gen_raw.shape[0])
        rmse = np.sqrt(np.mean((gt_raw[:T] - gen_raw[:T]) ** 2))

        safe_name = str(name)[:30].replace('/', '_')
        print(f"  [{idx_i+1}/{n}] {name} (T={T_len}, src={src}) RMSE={rmse:.4f}")
        print(f"    text: \"{text[:60]}...\"")

        path = os.path.join(output_root, f'{idx_i:03d}_{safe_name}.mp4')
        title = f'{name} [{src}] T={T_len}\n"{text[:50]}..."  RMSE={rmse:.4f}'
        save_comparison_video(gt_joints, gen_joints, path, title,
                              args.fps, args.viewport, 'GT', 'Generated')
        print(f"    saved: {path}")


if __name__ == '__main__':
    main()