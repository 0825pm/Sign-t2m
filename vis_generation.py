"""
vis_generation.py — Sign-t2m Text→Motion 생성 시각화

Usage:
    cd ~/Projects/research/sign-t2m

    # 120D — GT 비교 (val set, 3 datasets)
    python vis_generation.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --nfeats 120 --mode val \
        --dataset how2sign_csl_phoenix \
        --npy_root /home/user/Projects/research/SOKE/data/data360/How2Sign \
        --csl_npy_root /home/user/Projects/research/SOKE/data/data360/CSL-Daily \
        --phoenix_npy_root /home/user/Projects/research/SOKE/data/data360/Phoenix_2014T \
        --num_samples 10 --output vis_output/pos120

    # 120D — 자유 텍스트 생성 (multilingual)
    python vis_generation.py \
        --ckpt logs/.../checkpoints/last.ckpt \
        --nfeats 120 --mode text \
        --texts "a woman explains the weather forecast" "今天天气很好" "morgen wird es regnen" \
        --lengths 100 80 80 --output vis_output/pos120_text
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
# Constants — 44-joint skeleton
# =============================================================================
NFEATS = 528  # overridden by --nfeats

SPINE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4)]
BODY_CONNECTIONS = [
    (4, 7), (4, 5), (4, 6), (5, 8), (6, 9),
    (8, 10), (9, 11), (10, 12), (11, 13),
]

def _hand_conns(wrist, offset):
    c = []
    for f in range(5):
        b = offset + f * 3
        c += [(wrist, b), (b, b + 1), (b + 1, b + 2)]
    return c

LHAND_CONNECTIONS = _hand_conns(12, 14)
RHAND_CONNECTIONS = _hand_conns(13, 29)
ALL_CONNECTIONS   = SPINE_CONNECTIONS + BODY_CONNECTIONS + LHAND_CONNECTIONS + RHAND_CONNECTIONS

LHAND_INDICES = list(range(14, 29))
RHAND_INDICES = list(range(29, 44))

# 44-joint order → SMPLX 55 index 매핑
SEL_44_FROM_55 = (
    [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    + list(range(25, 40))
    + list(range(40, 55))
)

DEFAULT_SHAPE = np.array([
    -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
     0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
], dtype=np.float32)


# =============================================================================
# 120D → 44-joint via SMPLX FK
# =============================================================================

def feats120_to_joints44(features_raw, device='cuda:0'):
    """
    120D axis-angle → [T, 44, 3] joint positions via SMPLX FK

    120D layout:
      [0:30]   upper_body (10 joints x 3)
      [30:75]  lhand      (15 joints x 3)
      [75:120] rhand      (15 joints x 3)
    """
    from src.utils.human_models import get_coord

    T = features_raw.shape[0]

    upper_body = features_raw[:, 0:30]    # [T, 30]
    lhand      = features_raw[:, 30:75]   # [T, 45]
    rhand      = features_raw[:, 75:120]  # [T, 45]

    # body_pose: lower(33 zeros) + upper(30) = 63
    lower_zeros  = np.zeros((T, 33), dtype=np.float32)
    body_pose_np = np.concatenate([lower_zeros, upper_body], axis=-1)  # [T, 63]

    shape = np.tile(DEFAULT_SHAPE, (T, 1))  # [T, 10]

    def t(x):
        return torch.from_numpy(x).float().to(device)

    with torch.no_grad():
        _, joints_all = get_coord(
            t(np.zeros((T, 3),  dtype=np.float32)),   # root
            t(body_pose_np),
            t(lhand),
            t(rhand),
            t(np.zeros((T, 3),  dtype=np.float32)),   # jaw
            t(shape),
            t(np.zeros((T, 10), dtype=np.float32)),   # expr
        )
    joints55 = joints_all.cpu().numpy()  # [T, N, 3]

    # pelvis center
    joints55 = joints55 - joints55[:, 0:1, :]

    # 55 -> 44 select
    return joints55[:, SEL_44_FROM_55, :]  # [T, 44, 3]


# =============================================================================
# Feature -> 44 Joints  (nfeats 자동 분기)
# =============================================================================

def feats_to_joints(features_raw, nfeats=528, device='cuda:0'):
    """[T, D] -> [T, 44, 3]"""
    T = features_raw.shape[0]

    if nfeats == 120:
        try:
            return feats120_to_joints44(features_raw, device=device)
        except Exception as e:
            print(f"  WARNING: SMPLX FK failed ({e}), using approximate fallback")
            upper_body = features_raw[:, 0:30].reshape(T, 10, 3)
            lhand      = features_raw[:, 30:75].reshape(T, 15, 3)
            rhand      = features_raw[:, 75:120].reshape(T, 15, 3)
            pad        = np.zeros((T, 4, 3), dtype=np.float32)
            body14     = np.concatenate([upper_body, pad], axis=1)
            return np.concatenate([body14, lhand, rhand], axis=1)

    elif nfeats == 133:
        pelvis   = np.zeros((T, 1, 3), dtype=np.float32)
        body_ric = features_raw[:, 4:43].reshape(T, 13, 3)
        body_14  = np.concatenate([pelvis, body_ric], axis=1)
        lhand    = features_raw[:, 43:88].reshape(T, 15, 3) + body_14[:, 12:13, :]
        rhand    = features_raw[:, 88:133].reshape(T, 15, 3) + body_14[:, 13:14, :]
        return np.concatenate([body_14, lhand, rhand], axis=1)

    elif nfeats == 360:
        body_pos  = features_raw[:, 0:30].reshape(T, 10, 3)
        lhand_pos = features_raw[:, 90:135].reshape(T, 15, 3) + body_pos[:, 8:9, :]
        rhand_pos = features_raw[:, 225:270].reshape(T, 15, 3) + body_pos[:, 9:10, :]
        pad       = np.zeros((T, 4, 3), dtype=np.float32)
        body14    = np.concatenate([body_pos, pad], axis=1)
        return np.concatenate([body14, lhand_pos, rhand_pos], axis=1)

    else:  # 528
        return features_raw[:, :132].reshape(T, 44, 3)


# =============================================================================
# Skeleton Visualization
# =============================================================================

def _setup_skeleton_ax(ax, label, color, x_lim, y_lim):
    ax.set_title(label, fontsize=12, fontweight='bold', color=color)
    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.set_aspect('equal'); ax.axis('off')


def _get_viewport(data, viewport):
    if viewport > 0:
        return (-viewport, viewport), (-viewport, viewport)
    pts    = data.reshape(-1, 3)
    margin = 0.15
    vp     = max(pts[:, 0].max() - pts[:, 0].min(),
                 pts[:, 1].max() - pts[:, 1].min()) / 2 + margin
    vp     = max(vp, 0.3)
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
    spine_sc = ax.scatter([], [], c='purple',        s=20, zorder=5)
    body_sc  = ax.scatter([], [], c=colors['body'],  s=15, zorder=5)
    lhand_sc = ax.scatter([], [], c=colors['lhand'], s=5,  zorder=5)
    rhand_sc = ax.scatter([], [], c=colors['rhand'], s=5,  zorder=5)
    return lines, spine_sc, body_sc, lhand_sc, rhand_sc


def _update_skeleton(data, f, lines, spine_sc, body_sc, lhand_sc, rhand_sc):
    x, y = data[f, :, 0], -data[f, :, 1]
    for (line, i, j) in lines:
        line.set_data([x[i], x[j]], [y[i], y[j]])
    spine_sc.set_offsets(np.c_[x[:4],    y[:4]])
    body_sc.set_offsets(np.c_[x[4:14],   y[4:14]])
    lhand_sc.set_offsets(np.c_[x[14:29], y[14:29]])
    rhand_sc.set_offsets(np.c_[x[29:44], y[29:44]])


def _center_at_spine3(joints):
    return joints - joints[:, 3:4, :]


def save_comparison_video(left_joints, right_joints, save_path,
                          title='', fps=25, viewport=0,
                          left_label='GT', right_label='Generated'):
    T = min(left_joints.shape[0], right_joints.shape[0])
    left  = _center_at_spine3(left_joints[:T].copy())
    right = _center_at_spine3(right_joints[:T].copy())

    x_lim_l, y_lim_l = _get_viewport(left,  viewport)
    x_lim_r, y_lim_r = _get_viewport(right, viewport)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=10)
    _setup_skeleton_ax(ax_l, left_label,  'blue', x_lim_l, y_lim_l)
    _setup_skeleton_ax(ax_r, right_label, 'red',  x_lim_r, y_lim_r)

    colors = {'body': 'blue', 'lhand': 'red', 'rhand': 'green'}
    el_l = _build_skeleton_elements(ax_l, colors)
    el_r = _build_skeleton_elements(ax_r, colors)

    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T - 1)
        frame_text.set_text(f'Frame {f}/{T-1}')
        _update_skeleton(left,  f, *el_l)
        _update_skeleton(right, f, *el_r)
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


def save_single_video(joints, save_path, title='', fps=25, viewport=0):
    T    = joints.shape[0]
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

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(ckpt_path, device, guidance_scale=None, step_num=None):
    from src.models.sign_t2m import SignMotionGeneration
    from src.models.nets.sign_denoiser import SignDenoiser
    from diffusers import DDPMScheduler, UniPCMultistepScheduler

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt    = torch.load(ckpt_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})

    # prediction_type 자동 감지
    ns_config = hparams.get('noise_scheduler', None)
    if ns_config is not None and hasattr(ns_config, 'config'):
        pred_type = ns_config.config.get('prediction_type', 'sample')
    elif ns_config is not None and hasattr(ns_config, 'prediction_type'):
        pred_type = ns_config.prediction_type
    else:
        pred_type = 'sample'
    print(f"  prediction_type: {pred_type}")

    gs = guidance_scale if guidance_scale is not None else hparams.get('guidance_scale', 4.0)
    sn = step_num       if step_num       is not None else hparams.get('step_num', 10)

    # text encoder — checkpoint hparams에서 자동 감지
    te_cfg    = hparams.get('text_encoder', {})
    te_target = te_cfg.get('_target_', '') if isinstance(te_cfg, dict) else ''
    if 'MBart' in te_target:
        from src.models.nets.text_encoder import MBartTextEncoder
        model_path   = te_cfg.get('model_path',
            '/home/user/Projects/research/SOKE/deps/mbart-h2s-csl-phoenix')
        text_encoder = MBartTextEncoder(model_path=model_path, freeze_lm=True, output_dim=512)
    else:
        from src.models.nets.text_encoder import CLIP
        text_encoder = CLIP(freeze_lm=True)

    # denoiser
    dn_cfg     = hparams.get('denoiser', {})
    motion_dim = dn_cfg.get('motion_dim', NFEATS) if isinstance(dn_cfg, dict) else NFEATS
    stage_dim  = dn_cfg.get('stage_dim', '256*4') if isinstance(dn_cfg, dict) else '256*4'

    denoiser = SignDenoiser(
        motion_dim=motion_dim, max_motion_len=401, text_dim=512,
        pos_emb='cos', stage_dim=stage_dim, num_groups=16, patch_size=8,
        ssm_cfg={'d_state': 16, 'd_conv': 4, 'expand': 2},
        rms_norm=False, fused_add_norm=True, part_aware=False,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule='squaredcos_cap_v2', variance_type='fixed_small',
        clip_sample=False, prediction_type=pred_type,
    )
    # prediction_type 규칙:
    #   sample      → UniPC에 epsilon (sign_t2m.py에서 x0→eps 변환 후 전달)
    #   epsilon     → UniPC에 epsilon 그대로
    #   v_prediction → UniPC에 v_prediction 그대로
    unipc_pred_type = 'epsilon' if pred_type == 'sample' else pred_type
    print(f"  UniPC prediction_type: {unipc_pred_type}")
    sample_scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule='squaredcos_cap_v2', solver_order=2,
        prediction_type=unipc_pred_type, thresholding=False, sample_max_value=1.0,
    )

    ema_cfg = hparams.get('ema', {'use_ema': True, 'ema_decay': 0.999, 'ema_start': 1000})

    model = SignMotionGeneration(
        text_encoder=text_encoder, denoiser=denoiser,
        noise_scheduler=noise_scheduler, sample_scheduler=sample_scheduler,
        text_replace_prob=0.0, guidance_scale=gs,
        dataset_name=hparams.get('dataset_name', 'phoenix'),
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
        lr_scheduler=None, step_num=sn, ema=ema_cfg,
    )

    state   = ckpt['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_te  = [k for k in missing if not k.startswith('text_encoder')]
    if non_te:
        print(f"  WARNING Missing keys (non-text_encoder): {non_te}")
    if unexpected:
        print(f"  WARNING Unexpected keys: {unexpected[:5]}")

    model.eval().to(device)
    n_params = sum(p.numel() for p in model.denoiser.parameters())
    print(f"  Epoch: {ckpt.get('epoch','?')}, Step: {ckpt.get('global_step','?')}, "
          f"Denoiser: {n_params/1e6:.2f}M")
    print(f"  guidance_scale: {gs}, step_num: {sn}")
    return model


# =============================================================================
# Text mode
# =============================================================================

def _run_text_mode(model, args, device, mean_np, std_np, output_root):
    texts   = args.texts   or ["a person waves hello"]
    lengths = args.lengths or [100] * len(texts)
    assert len(texts) == len(lengths), "--texts와 --lengths 개수가 달라요"

    print(f"\n[2/3] Text mode: {len(texts)} prompts")
    # dataset에서 src 결정
    ds = args.dataset
    if 'phoenix' in ds:
        default_src = 'phoenix'
    elif 'csl' in ds:
        default_src = 'csl'
    else:
        default_src = 'how2sign'

    for i, (text, length) in enumerate(zip(texts, lengths)):
        src = default_src
        print(f"\n  [{i+1}/{len(texts)}] \"{text}\" (T={length}, src={src})")

        with torch.no_grad():
            generated = model.generate([text], [length], srcs=[src])
        gen_np  = generated[0].cpu().numpy()
        gen_raw = gen_np * np.maximum(std_np, 0.01) + mean_np
        joints  = feats_to_joints(gen_raw, NFEATS, device=str(device))
        print(f"    output range: [{gen_raw.min():.3f}, {gen_raw.max():.3f}]")
        safe_text = text[:40].replace(' ', '_').replace('/', '_')
        path = os.path.join(output_root, f'{i:03d}_{safe_text}.mp4')
        save_single_video(joints, path, f'"{text}"\nT={length}', args.fps, args.viewport)
        print(f"    saved: {path}")


# =============================================================================
# Val mode
# =============================================================================

def _run_val_mode(model, args, device, mean, std, mean_np, std_np,
                  csl_mean, csl_std, output_root):
    print(f"\n[2/3] Val mode: {args.dataset} / {args.split}")

    from src.data.signlang.dataset_sign import SignText2MotionDataset

    dataset = SignText2MotionDataset(
        data_root=args.data_root, csl_root=args.csl_root,
        phoenix_root=args.phoenix_root, npy_root=args.npy_root,
        csl_npy_root=args.csl_npy_root, phoenix_npy_root=args.phoenix_npy_root,
        split=args.split,
        mean=mean, std=std,
        csl_mean=csl_mean, csl_std=csl_std,
        nfeats=NFEATS, dataset_name=args.dataset,
        max_motion_length=400, min_motion_length=20,
    )

    n       = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    print(f"  Dataset: {len(dataset)} samples, visualizing {n}")
    print(f"\n[3/3] Generating...")

    for idx_i, ds_idx in enumerate(indices):
        item = dataset[ds_idx]
        if item is None:
            continue

        gt_norm = item['motion'].numpy()
        text    = item['text']
        name    = item['name']
        src     = item.get('src', 'how2sign')
        T_len   = item['motion_len']

        # GT 역정규화
        gt_raw = gt_norm * np.maximum(std_np, 0.01) + mean_np

        # 생성 → 역정규화
        with torch.no_grad():
            generated = model.generate([text], [T_len], srcs=[src])
        gen_np  = generated[0].cpu().numpy()
        gen_raw = gen_np * np.maximum(std_np, 0.01) + mean_np

        gt_joints  = feats_to_joints(gt_raw,  NFEATS, device=str(device))
        gen_joints = feats_to_joints(gen_raw, NFEATS, device=str(device))

        T    = min(gt_raw.shape[0], gen_raw.shape[0])
        rmse = np.sqrt(np.mean((gt_raw[:T] - gen_raw[:T]) ** 2))

        safe_name = str(name)[:30].replace('/', '_')
        print(f"  [{idx_i+1}/{n}] {name} (T={T_len}, src={src}) RMSE={rmse:.4f}")
        print(f"    text: \"{text[:70]}\"")

        path  = os.path.join(output_root, f'{idx_i:03d}_{safe_name}.mp4')
        title = f'{name} [{src}] T={T_len}\n"{text[:50]}..."  RMSE={rmse:.4f}'
        save_comparison_video(gt_joints, gen_joints, path, title,
                              args.fps, args.viewport)
        print(f"    saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sign-t2m Generation Visualization')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--nfeats', type=int, default=120,
                        choices=[120, 133, 360, 528])
    parser.add_argument('--mode', default='val', choices=['val', 'text'])
    # data paths
    parser.add_argument('--data_root',
        default='/home/user/Projects/research/SOKE/data/How2Sign')
    parser.add_argument('--csl_root',
        default='/home/user/Projects/research/SOKE/data/CSL-Daily')
    parser.add_argument('--phoenix_root',
        default='/home/user/Projects/research/SOKE/data/Phoenix_2014T')
    parser.add_argument('--npy_root',         default=None)
    parser.add_argument('--csl_npy_root',     default=None)
    parser.add_argument('--phoenix_npy_root', default=None)
    parser.add_argument('--mean_path',
        default='/home/user/Projects/research/SOKE/data/data360/How2Sign/mean_pos120.pt')
    parser.add_argument('--std_path',
        default='/home/user/Projects/research/SOKE/data/data360/How2Sign/std_pos120.pt')
    parser.add_argument('--csl_mean_path',
        default='/home/user/Projects/research/SOKE/data/data360/CSL-Daily/mean_pos120.pt')
    parser.add_argument('--csl_std_path',
        default='/home/user/Projects/research/SOKE/data/data360/CSL-Daily/std_pos120.pt')
    parser.add_argument('--dataset',     default='how2sign_csl_phoenix')
    parser.add_argument('--split',       default='val')
    parser.add_argument('--num_samples', type=int, default=5)
    # text mode
    parser.add_argument('--texts',   nargs='+', default=None)
    parser.add_argument('--lengths', nargs='+', type=int, default=None)
    # generation
    parser.add_argument('--guidance_scale', type=float, default=None)
    parser.add_argument('--step_num',       type=int,   default=None)
    # vis
    parser.add_argument('--fps',      type=int,   default=25)
    parser.add_argument('--viewport', type=float, default=0)
    parser.add_argument('--output',   default='vis_generation_output')
    parser.add_argument('--device',   default='cuda:0')
    args = parser.parse_args()

    global NFEATS
    NFEATS = args.nfeats

    device      = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'gen_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("Sign-t2m Generation Visualization")
    print("=" * 60)

    print("\n[1/3] Loading model...")
    model = load_model(args.ckpt, device, args.guidance_scale, args.step_num)

    mean    = torch.load(args.mean_path, map_location='cpu').float()[:NFEATS]
    std     = torch.load(args.std_path,  map_location='cpu').float()[:NFEATS]
    mean_np = mean.numpy()
    std_np  = std.numpy()

    csl_mean, csl_std = mean, std
    if args.csl_mean_path and os.path.exists(args.csl_mean_path):
        csl_mean = torch.load(args.csl_mean_path, map_location='cpu').float()[:NFEATS]
    if args.csl_std_path and os.path.exists(args.csl_std_path):
        csl_std  = torch.load(args.csl_std_path,  map_location='cpu').float()[:NFEATS]

    if args.mode == 'text':
        _run_text_mode(model, args, device, mean_np, std_np, output_root)
    else:
        _run_val_mode(model, args, device, mean, std, mean_np, std_np,
                      csl_mean, csl_std, output_root)

    print(f"\n{'='*60}")
    print(f"Done. Videos saved to {output_root}")
    print("="*60)


if __name__ == '__main__':
    main()
