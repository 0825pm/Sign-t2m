"""
vis_generation_528d.py — Sign-t2m 528D Text→Motion 생성 시각화

528D 모델 체크포인트에서 text → motion 생성 후 skeleton 영상 저장
528D = positions(132) + velocities(132) + 6D_rot(264), 44 joints
positions를 직접 사용하므로 FK 불필요

Usage:
    cd ~/Projects/research/Sign-t2m

    # GT 비교 (val set에서 N개 샘플 → GT vs Generated)
    python vis_generation_528d.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --mode val --num_samples 5

    # 자유 텍스트 생성
    python vis_generation_528d.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --mode text \
        --texts "a person waves hello" "pointing to the right" \
        --lengths 80 60
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
# Constants — 44-joint skeleton (528D)
# =============================================================================
N_JOINTS = 44
NFEATS = 528  # pos(132) + vel(132) + 6d(264)

# Joint layout:
#   0-3:   Pelvis, Spine1, Spine2, Spine3
#   4-13:  Neck, L_Collar, R_Collar, Head, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist
#   14-28: left hand (15)
#   29-43: right hand (15)

SPINE_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4)]
BODY_CONNECTIONS = [
    (4,7), (4,5), (4,6), (5,8), (6,9),
    (8,10), (9,11), (10,12), (11,13),
]

def _hand_conns(wrist, offset):
    c = []
    for f in range(5):
        b = offset + f*3
        c += [(wrist, b), (b, b+1), (b+1, b+2)]
    return c

LHAND_CONNECTIONS = _hand_conns(12, 14)
RHAND_CONNECTIONS = _hand_conns(13, 29)
ALL_CONNECTIONS = SPINE_CONNECTIONS + BODY_CONNECTIONS + LHAND_CONNECTIONS + RHAND_CONNECTIONS

BODY_INDICES = list(range(14))
LHAND_INDICES = list(range(14, 29))
RHAND_INDICES = list(range(29, 44))


# =============================================================================
# Skeleton Visualization (44-joint)
# =============================================================================

def _setup_skeleton_ax(ax, label, color, x_lim, y_lim):
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.axis('off')


def _get_viewport(data, viewport):
    """data: [T, 44, 3]"""
    if viewport > 0:
        return (-viewport, viewport), (-viewport, viewport)
    pts = data.reshape(-1, 3)
    margin = 0.15
    xr = pts[:, 0].max() - pts[:, 0].min()
    yr = pts[:, 1].max() - pts[:, 1].min()
    vp = max(xr, yr) / 2 + margin
    vp = max(vp, 0.3)
    return (-vp, vp), (-vp, vp)


def _build_skeleton_elements(ax, colors):
    lines = []
    for (i, j) in ALL_CONNECTIONS:
        if i in LHAND_INDICES or j in LHAND_INDICES:
            c, lw = colors['lhand'], 0.8
        elif i in RHAND_INDICES or j in RHAND_INDICES:
            c, lw = colors['rhand'], 0.8
        elif (i, j) in SPINE_CONNECTIONS:
            c, lw = 'purple', 2.5
        else:
            c, lw = colors['body'], 2.0
        line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
        lines.append((line, i, j))
    spine_sc = ax.scatter([], [], c='purple', s=20, zorder=5)
    body_sc = ax.scatter([], [], c=colors['body'], s=15, zorder=5)
    lhand_sc = ax.scatter([], [], c=colors['lhand'], s=5, zorder=5)
    rhand_sc = ax.scatter([], [], c=colors['rhand'], s=5, zorder=5)
    return lines, spine_sc, body_sc, lhand_sc, rhand_sc


def _update_skeleton(data, f, lines, spine_sc, body_sc, lhand_sc, rhand_sc):
    x, y = data[f, :, 0], -data[f, :, 1]
    for (line, i, j) in lines:
        line.set_data([x[i], x[j]], [y[i], y[j]])
    spine_sc.set_offsets(np.c_[x[:4], y[:4]])
    body_sc.set_offsets(np.c_[x[4:14], y[4:14]])
    lhand_sc.set_offsets(np.c_[x[14:29], y[14:29]])
    rhand_sc.set_offsets(np.c_[x[29:44], y[29:44]])


def _center_at_spine3(joints):
    """Center at Spine3 (idx 3)"""
    return joints - joints[:, 3:4, :]


def save_comparison_video(left_joints, right_joints, save_path,
                          title='', fps=25, viewport=0.5,
                          left_label='GT', right_label='Generated'):
    T = min(left_joints.shape[0], right_joints.shape[0])
    left = _center_at_spine3(left_joints[:T].copy())
    right = _center_at_spine3(right_joints[:T].copy())

    all_data = np.concatenate([left, right], axis=0)
    # 각각 독립 viewport — Generated 폭발해도 GT 정상 표시
    x_lim_l, y_lim_l = _get_viewport(left, viewport)
    x_lim_r, y_lim_r = _get_viewport(right, viewport)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=10)
    _setup_skeleton_ax(ax_l, left_label, 'blue', x_lim_l, y_lim_l)
    _setup_skeleton_ax(ax_r, right_label, 'red', x_lim_r, y_lim_r)

    colors = {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}
    el_l = _build_skeleton_elements(ax_l, colors)
    el_r = _build_skeleton_elements(ax_r, colors)

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        _update_skeleton(left, f, *el_l)
        _update_skeleton(right, f, *el_r)
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


def save_single_video(joints, save_path, title='', fps=25, viewport=0.5):
    T = joints.shape[0]
    data = _center_at_spine3(joints.copy())
    x_lim, y_lim = _get_viewport(data, viewport)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(title, fontsize=10)
    _setup_skeleton_ax(ax, 'Generated', 'red', x_lim, y_lim)

    colors = {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}
    el = _build_skeleton_elements(ax, colors)

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        _update_skeleton(data, f, *el)
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# 528D Feature → 44 Joints (직접 추출, FK 불필요!)
# =============================================================================

def feats_to_joints(features_raw, nfeats=528):
    """feature → [T, 44, 3]"""
    T = features_raw.shape[0]
    if nfeats == 133:
        # 133D: root_motion[0:4] + body_ric[4:43] + lhand_ric[43:88] + rhand_ric[88:133]
        # position 직접 사용, FK 불필요
        pelvis   = np.zeros((T, 1, 3), dtype=np.float32)
        body_ric = features_raw[:, 4:43].reshape(T, 13, 3)
        body_14  = np.concatenate([pelvis, body_ric], axis=1)  # [T,14,3]
        lhand    = features_raw[:, 43:88].reshape(T, 15, 3) + body_14[:, 12:13, :]
        rhand    = features_raw[:, 88:133].reshape(T, 15, 3) + body_14[:, 13:14, :]
        return np.concatenate([body_14, lhand, rhand], axis=1)  # [T,44,3]
    elif nfeats == 360:
        # 360D: body_pos[0:30] + lhand_pos[90:135] + rhand_pos[225:270]
        body_pos  = features_raw[:, 0:30].reshape(T, 10, 3)
        lhand_pos = features_raw[:, 90:135].reshape(T, 15, 3) + body_pos[:, 8:9, :]
        rhand_pos = features_raw[:, 225:270].reshape(T, 15, 3) + body_pos[:, 9:10, :]
        # body를 14-joint으로 padding (skeleton 호환)
        pad = np.zeros((T, 4, 3), dtype=np.float32)
        body_14 = np.concatenate([body_pos, pad], axis=1)
        return np.concatenate([body_14, lhand_pos, rhand_pos], axis=1)  # [T,44,3]
    else:
        # 528D: positions 처음 132D
        return features_raw[:, :132].reshape(T, 44, 3)


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

    # Auto-detect prediction_type from checkpoint
    ns_config = hparams.get('noise_scheduler', None)
    if ns_config is not None and hasattr(ns_config, 'config'):
        pred_type = ns_config.config.get('prediction_type', 'epsilon')
    elif ns_config is not None and hasattr(ns_config, 'prediction_type'):
        pred_type = ns_config.prediction_type
    else:
        pred_type = 'epsilon'
    print(f"  prediction_type from checkpoint: '{pred_type}'")

    # Build components
    text_encoder = CLIP(freeze_lm=True)

    # Auto-detect motion_dim and stage_dim from checkpoint weights
    sd = ckpt['state_dict']
    motion_dim = sd['denoiser.m_input_proj.weight'].shape[1]   # [stage_dim, motion_dim]
    stage_dim  = sd['denoiser.m_input_proj.weight'].shape[0]   # [stage_dim, motion_dim]
    print(f"  motion_dim from checkpoint: {motion_dim}")
    print(f"  stage_dim  from checkpoint: {stage_dim}")

    denoiser = SignDenoiser(
        motion_dim=motion_dim,
        max_motion_len=401,
        text_dim=512,
        pos_emb="cos",
        stage_dim=f"{stage_dim}*4",
        num_groups=16,
        patch_size=8,
        ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
        rms_norm=False,
        fused_add_norm=True,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
        clip_sample=False, prediction_type=pred_type,
    )

    sample_scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", solver_order=2,
        prediction_type='epsilon',  # FINDINGS.md: 항상 epsilon 하드코딩!
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
    parser.add_argument('--nfeats', type=int, default=133, choices=[133, 360, 528])
    # Mode
    parser.add_argument('--mode', default='val', choices=['val', 'text'],
                        help='val: GT comparison / text: free generation')
    # Data (for val mode)
    parser.add_argument('--data_root', default='/home/user/Projects/research/SOKE/data/How2Sign')
    parser.add_argument('--csl_root', default='/home/user/Projects/research/SOKE/data/CSL-Daily')
    parser.add_argument('--phoenix_root', default='/home/user/Projects/research/SOKE/data/Phoenix_2014T')
    parser.add_argument('--npy_root', default='/home/user/Projects/research/SOKE/data/How2Sign_528d')
    parser.add_argument('--csl_npy_root', default='/home/user/Projects/research/SOKE/data/CSL-Daily_528d')
    parser.add_argument('--phoenix_npy_root', default='/home/user/Projects/research/SOKE/data/Phoenix_528d')
    parser.add_argument('--mean_path', default='/home/user/Projects/research/SOKE/data/How2Sign_528d/mean_528.pt')
    parser.add_argument('--std_path', default='/home/user/Projects/research/SOKE/data/How2Sign_528d/std_528.pt')
    parser.add_argument('--csl_mean_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily_528d/mean_528.pt')
    parser.add_argument('--csl_std_path', default='/home/user/Projects/research/SOKE/data/CSL-Daily_528d/std_528.pt')
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
    parser.add_argument('--viewport', type=float, default=0)  # 0=auto
    parser.add_argument('--output', default='vis_generation_output')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    global NFEATS
    NFEATS = args.nfeats

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
    mean = torch.load(args.mean_path, map_location='cpu').float()[:NFEATS]
    std = torch.load(args.std_path, map_location='cpu').float()[:NFEATS]
    mean_np, std_np = mean.numpy(), std.numpy()

    csl_mean, csl_std = mean, std
    if os.path.exists(args.csl_mean_path):
        csl_mean = torch.load(args.csl_mean_path, map_location='cpu').float()[:NFEATS]
    if os.path.exists(args.csl_std_path):
        csl_std = torch.load(args.csl_std_path, map_location='cpu').float()[:NFEATS]

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
            generated = model.generate([text], [length])  # [1, T, 528]

        gen_np = generated[0].cpu().numpy()  # [T, 528]
        gen_raw = gen_np * (std_np + 1e-10) + mean_np

        joints = feats_to_joints(gen_raw, NFEATS)

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
        npy_root=args.npy_root,
        csl_npy_root=args.csl_npy_root,
        phoenix_npy_root=args.phoenix_npy_root,
        split=args.split,
        mean=mean,
        std=std,
        csl_mean=csl_mean,
        csl_std=csl_std,
        nfeats=NFEATS,
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

        gt_norm = item['motion'].numpy()           # [T, 528] normalized
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
            generated = model.generate([text], [T_len])  # [1, T, 528]
        gen_np = generated[0].cpu().numpy()
        gen_raw = gen_np * (std_np + 1e-10) + mean_np

        # Joints — 528D 처음 132D가 positions, FK 불필요
        gt_joints = feats_to_joints(gt_raw, NFEATS)
        gen_joints = feats_to_joints(gen_raw, NFEATS)

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