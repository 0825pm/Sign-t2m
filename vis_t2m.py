"""
vis_t2m.py — Sign-t2m Text→Motion 시각화

train.py와 동일한 데이터로더 사용.
523D/528D 자동 감지.
렌더링: vis_528d_quick.py / vis_523d_quick.py 그대로.

Usage:
    cd ~/Projects/research/sign-t2m
    python vis_t2m.py --ckpt logs/.../last.ckpt --split val --num_samples 5
    python vis_t2m.py --ckpt logs/.../last.ckpt --split train --num_samples 5
"""

import os, sys, argparse
import numpy as np
import torch
from datetime import datetime
from functools import partial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.backends.cudnn.enabled = False

# =============================================================================
# 44-joint skeleton (vis_528d_quick.py 복붙)
# =============================================================================
SPINE = [(0,1),(1,2),(2,3),(3,4)]
BODY = [(4,7),(4,5),(4,6),(5,8),(6,9),(8,10),(9,11),(10,12),(11,13)]
def _hand(wrist, offset):
    c = []
    for f in range(5):
        b = offset + f*3
        c += [(wrist, b), (b, b+1), (b+1, b+2)]
    return c
LHAND = _hand(12, 14)
RHAND = _hand(13, 29)
ALL_CONN = SPINE + BODY + LHAND + RHAND

# =============================================================================
# Rendering (vis_528d_quick.py 그대로)
# =============================================================================
def _render_setup(ax, joints, viewport):
    centered = joints - joints[:, 3:4, :]
    x_all = centered[:, :, 0]
    y_all = -centered[:, :, 1]
    if viewport > 0:
        cx, cy, vp = 0, 0, viewport
    else:
        margin = 0.05
        xmin, xmax = x_all.min() - margin, x_all.max() + margin
        ymin, ymax = y_all.min() - margin, y_all.max() + margin
        vp = max(xmax - xmin, ymax - ymin) / 2
        vp = max(vp, 0.15)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
    ax.set_xlim(cx - vp, cx + vp)
    ax.set_ylim(cy - vp, cy + vp)
    ax.set_aspect('equal')
    ax.axis('off')
    lines = []
    for (i, j) in ALL_CONN:
        if i >= 14 or j >= 14:
            c, lw = ('#E91E63' if (i < 29 and j < 29) else '#4CAF50'), 0.8
        elif (i,j) in SPINE:
            c, lw = 'purple', 2.5
        else:
            c, lw = '#333333', 2.0
        line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
        lines.append((line, i, j))
    body_sc = ax.scatter([], [], c='#333333', s=15, zorder=5)
    lhand_sc = ax.scatter([], [], c='#E91E63', s=5, zorder=5)
    rhand_sc = ax.scatter([], [], c='#4CAF50', s=5, zorder=5)
    return centered, lines, body_sc, lhand_sc, rhand_sc

def _render_update(centered, f, lines, body_sc, lhand_sc, rhand_sc):
    x = centered[f, :, 0]
    y = -centered[f, :, 1]
    for (line, i, j) in lines:
        line.set_data([x[i], x[j]], [y[i], y[j]])
    body_sc.set_offsets(np.c_[x[:14], y[:14]])
    lhand_sc.set_offsets(np.c_[x[14:29], y[14:29]])
    rhand_sc.set_offsets(np.c_[x[29:44], y[29:44]])

def save_comparison_video(gt_joints, gen_joints, save_path,
                          title='', fps=25, viewport=0):
    T = min(gt_joints.shape[0], gen_joints.shape[0])
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=9)
    ax_l.set_title('GT', fontsize=12, fontweight='bold', color='blue')
    ax_r.set_title('Generated', fontsize=12, fontweight='bold', color='red')
    gt_c, *el_l = _render_setup(ax_l, gt_joints[:T], viewport)
    gen_c, *el_r = _render_setup(ax_r, gen_joints[:T], viewport)
    frame_txt = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    def update(f):
        frame_txt.set_text(f'Frame {f}/{T-1}')
        _render_update(gt_c, f, *el_l)
        _render_update(gen_c, f, *el_r)
        return []
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)

# =============================================================================
# denorm → joints (133D/523D/528D 자동 감지)
# =============================================================================
def to_joints_528(raw):
    """528D: positions[:132] → [T, 44, 3]"""
    return raw[:, :132].reshape(-1, 44, 3)

def to_joints_523(raw):
    """523D: body_ric + hand_ric(wrist-relative) → [T, 44, 3]"""
    T = raw.shape[0]
    pelvis = np.zeros((T, 1, 3), dtype=np.float32)
    body_ric = raw[:, 4:43].reshape(T, 13, 3)
    body_14 = np.concatenate([pelvis, body_ric], axis=1)
    lwrist = body_14[:, 12:13, :]
    rwrist = body_14[:, 13:14, :]
    lhand = raw[:, 163:208].reshape(T, 15, 3) + lwrist
    rhand = raw[:, 343:388].reshape(T, 15, 3) + rwrist
    return np.concatenate([body_14, lhand, rhand], axis=1)

def to_joints_133(raw):
    """133D: root(4) + body_ric(39) + lhand_ric(45) + rhand_ric(45) → [T, 44, 3]"""
    T = raw.shape[0]
    pelvis = np.zeros((T, 1, 3), dtype=np.float32)
    body_ric = raw[:, 4:43].reshape(T, 13, 3)
    body_14 = np.concatenate([pelvis, body_ric], axis=1)
    lwrist = body_14[:, 12:13, :]
    rwrist = body_14[:, 13:14, :]
    lhand = raw[:, 43:88].reshape(T, 15, 3) + lwrist
    rhand = raw[:, 88:133].reshape(T, 15, 3) + rwrist
    return np.concatenate([body_14, lhand, rhand], axis=1)

def to_joints(feats_norm, mean_np, std_np):
    raw = feats_norm * (std_np + 1e-10) + mean_np
    D = raw.shape[-1]
    if D == 133:
        # 133D = axis-angle (SOKE format) → SMPLX FK
        try:
            from src.utils.feats2joints import feats2joints_smplx
            import torch
            mean_t = torch.from_numpy(mean_np).float()
            std_t  = torch.from_numpy(std_np).float()
            feat_t = torch.from_numpy(feats_norm).float().unsqueeze(0)  # [1,T,133]
            _, joints = feats2joints_smplx(feat_t, mean_t, std_t)
            joints_np = joints[0].cpu().numpy()  # [T, 144, 3]
            # 44 joints만 사용 (body14 + lhand15 + rhand15)
            # SMPLX joint indices: pelvis=0, body1-13, lwrist=20, rwrist=21, lhand=25-39, rhand=40-54
            BODY_IDX  = list(range(0, 14))
            LHAND_IDX = list(range(25, 40))
            RHAND_IDX = list(range(40, 55))
            sel = BODY_IDX + LHAND_IDX + RHAND_IDX
            return joints_np[:, sel, :]  # [T, 44, 3]
        except Exception as e:
            print(f"[vis_t2m] SMPLX FK failed ({e}), fallback to approx")
            return _joints_133_approx(raw)
    elif D == 523:
        return to_joints_523(raw)
    else:
        return to_joints_528(raw)


def _joints_133_approx(raw):
    """SMPLX 없을 때 133D axis-angle → 44-joint 근사 (시각화만 가능)"""
    T = raw.shape[0]
    joints = np.zeros((T, 44, 3), dtype=np.float32)
    # axis-angle을 position으로 직접 쓰는 건 불가, 0으로 채움 (경고용 fallback)
    return joints

# =============================================================================
# Model Loading
# =============================================================================
def load_model(ckpt_path, device, guidance_scale=None, step_num=None):
    from src.models.sign_t2m import SignMotionGeneration
    from src.models.nets.sign_denoiser import SignDenoiser
    from src.models.nets.text_encoder import CLIP
    from diffusers import DDPMScheduler, UniPCMultistepScheduler

    ckpt = torch.load(ckpt_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})
    state = ckpt['state_dict']

    ns = hparams.get('noise_scheduler', None)
    if ns is not None and hasattr(ns, 'config'):
        pred_type = ns.config.get('prediction_type', 'epsilon')
    elif ns is not None and hasattr(ns, 'prediction_type'):
        pred_type = ns.prediction_type
    else:
        pred_type = 'epsilon'

    stage_keys = sorted([k for k in state if 'layers.' in k and 'input_proj.weight' in k])
    if stage_keys:
        stage_dim = '*'.join(str(state[k].shape[0]) for k in stage_keys)
    else:
        # Detect from layer conv weights
        layer_dims = []
        for i in range(10):
            k = f'denoiser.layers.{i}.local_module1.conv.0.weight'
            if k in state:
                layer_dims.append(str(state[k].shape[0]))
        stage_dim = '*'.join(layer_dims) if layer_dims else "256*4"
    m_proj_key = 'denoiser.m_input_proj.weight'
    motion_dim = state[m_proj_key].shape[1] if m_proj_key in state else 528

    print(f"  pred='{pred_type}', stage='{stage_dim}', dim={motion_dim}")

    text_encoder = CLIP(freeze_lm=True)
    denoiser = SignDenoiser(
        motion_dim=motion_dim, max_motion_len=401, text_dim=512,
        pos_emb="cos", stage_dim=stage_dim, num_groups=16, patch_size=8,
        ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
        rms_norm=False, fused_add_norm=True,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
        clip_sample=False, prediction_type=pred_type,
    )
    sample_scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", solver_order=2,
        prediction_type=pred_type,
    )

    gs = guidance_scale or hparams.get('guidance_scale', 7.5)
    sn = step_num or hparams.get('step_num', 10)

    model = SignMotionGeneration(
        text_encoder=text_encoder, denoiser=denoiser,
        noise_scheduler=noise_scheduler, sample_scheduler=sample_scheduler,
        text_replace_prob=0.0, guidance_scale=gs,
        dataset_name=hparams.get('dataset_name', 'how2sign'),
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
        lr_scheduler=None, step_num=sn,
        ema=hparams.get('ema', {"use_ema": False, "ema_decay": 0.99, "ema_start": 1000}),
    )
    missing, _ = model.load_state_dict(state, strict=False)
    non_te = [k for k in missing if not k.startswith('text_encoder')]
    if non_te:
        print(f"  ⚠️  Missing: {non_te}")
    model.eval().to(device)
    print(f"  Epoch={ckpt.get('epoch','?')}, Step={ckpt.get('global_step','?')}, gs={gs}, steps={sn}")
    return model, hparams

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--guidance_scale', type=float, default=None)
    parser.add_argument('--step_num', type=int, default=None)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0, help='0=auto, >0=fixed')
    parser.add_argument('--output', default='vis_t2m_output')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output, f'{args.split}_{timestamp}')
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("Sign-t2m Text→Motion Visualization")
    print("=" * 60)

    # 1. Load model
    print("\n[1/3] Loading model...")
    model, _ = load_model(args.ckpt, device, args.guidance_scale, args.step_num)

    # 2. Auto-detect mean/std from dataset_name + motion_dim
    BASE = '/home/user/Projects/research/SOKE/data'
    ckpt_data = torch.load(args.ckpt, map_location='cpu')
    hp = ckpt_data.get('hyper_parameters', {})
    dataset_name = hp.get('dataset_name', 'how2sign')
    motion_dim = 528
    m_proj_key = 'denoiser.m_input_proj.weight'
    ckpt_state = ckpt_data.get('state_dict', {})
    if m_proj_key in ckpt_state:
        motion_dim = ckpt_state[m_proj_key].shape[1]
    del ckpt_data  # free memory

    if motion_dim == 133:
        suffix, stat_name = '133d', '133'
    elif motion_dim == 523:
        suffix, stat_name = '523d', '523'
    else:
        suffix, stat_name = '528d', '528'

    paths = {
        'how2sign': (f'{BASE}/How2Sign_{suffix}/mean_{stat_name}.pt', f'{BASE}/How2Sign_{suffix}/std_{stat_name}.pt'),
        'phoenix':  (f'{BASE}/Phoenix_{suffix}/mean_{stat_name}.pt',  f'{BASE}/Phoenix_{suffix}/std_{stat_name}.pt'),
        'csl':      (f'{BASE}/CSL-Daily_{suffix}/mean_{stat_name}.pt', f'{BASE}/CSL-Daily_{suffix}/std_{stat_name}.pt'),
    }
    mean_path, std_path = paths.get(dataset_name, paths['how2sign'])

    D = motion_dim
    mean = torch.load(mean_path, map_location='cpu').float()[:D]
    std = torch.load(std_path, map_location='cpu').float()[:D]
    mean_np, std_np = mean.numpy(), std.numpy()

    # 3. Load dataset (train.py와 동일)
    print(f"\n[2/3] Loading {dataset_name} / {args.split}...")
    from src.data.signlang.dataset_sign import SignText2MotionDataset
    dataset = SignText2MotionDataset(
        data_root=f'{BASE}/How2Sign', split=args.split,
        mean=mean, std=std, nfeats=D, dataset_name=dataset_name,
        max_motion_length=400, min_motion_length=20,
        csl_root=f'{BASE}/CSL-Daily',
        phoenix_root=f'{BASE}/Phoenix_2014T',
        npy_root=f'{BASE}/How2Sign_{suffix}',
        csl_npy_root=f'{BASE}/CSL-Daily_{suffix}',
        phoenix_npy_root=f'{BASE}/Phoenix_{suffix}',
        csl_mean=mean, csl_std=std,
    )

    n = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    print(f"  Dataset: {len(dataset)} samples, visualizing {n}")

    # 4. Generate & Visualize
    print(f"\n[3/3] Generating ({args.split})...\n")
    rmse_list, body_list, hand_list = [], [], []

    for idx_i, ds_idx in enumerate(indices):
        item = dataset[ds_idx]
        if item is None:
            continue

        gt_norm = item['motion']        # [T, D] normalized
        text = item['text']
        name = item.get('name', f'sample_{ds_idx}')
        T_len = int(item['motion_len'])

        # GT: denorm → joints
        gt_joints = to_joints(gt_norm.numpy(), mean_np, std_np)

        # Generated: model → denorm → joints
        with torch.no_grad():
            gt_tensor = gt_norm.unsqueeze(0).float().to(device)
            length = torch.tensor([T_len], device=device)
            gen_tensor = model.sample_motion(gt_tensor, length, [text])
            gen_np = gen_tensor[0].cpu().numpy()
        gen_joints = to_joints(gen_np, mean_np, std_np)

        T = min(gt_joints.shape[0], gen_joints.shape[0])
        gt_joints, gen_joints = gt_joints[:T], gen_joints[:T]

        # Metrics
        diff = gt_joints - gen_joints
        rmse = np.sqrt(np.mean(diff ** 2))
        body_rmse = np.sqrt(np.mean(diff[:, :14] ** 2))
        hand_rmse = np.sqrt(np.mean(diff[:, 14:44] ** 2))
        rmse_list.append(rmse)
        body_list.append(body_rmse)
        hand_list.append(hand_rmse)

        safe_name = str(name)[:30].replace('/', '_').replace(' ', '_')
        print(f"  [{idx_i+1}/{n}] {name} (T={T_len})")
        print(f"    text: \"{text[:60]}\"")
        print(f"    RMSE: total={rmse:.4f}  body={body_rmse:.4f}  hand={hand_rmse:.4f}")

        path = os.path.join(output_root, f'{idx_i:03d}_{safe_name}.mp4')
        title = (f'{name[:40]} T={T_len}\n'
                 f'"{text[:50]}"\n'
                 f'RMSE={rmse:.4f} (body={body_rmse:.4f} hand={hand_rmse:.4f})')
        save_comparison_video(gt_joints, gen_joints, path, title, args.fps, args.viewport)
        print(f"    → {path}")

    # Summary
    if rmse_list:
        print(f"\n{'='*60}")
        print(f"Summary ({len(rmse_list)} samples, {args.split})")
        print(f"  RMSE:      {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
        print(f"  Body RMSE: {np.mean(body_list):.4f} ± {np.std(body_list):.4f}")
        print(f"  Hand RMSE: {np.mean(hand_list):.4f} ± {np.std(hand_list):.4f}")
        print(f"\nVideos: {output_root}")
        print("=" * 60)

        with open(os.path.join(output_root, 'summary.txt'), 'w') as f:
            f.write(f"Checkpoint: {args.ckpt}\n")
            f.write(f"Split: {args.split}, Samples: {len(rmse_list)}, Dataset: {dataset_name}\n\n")
            f.write(f"RMSE:      {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}\n")
            f.write(f"Body RMSE: {np.mean(body_list):.4f} ± {np.std(body_list):.4f}\n")
            f.write(f"Hand RMSE: {np.mean(hand_list):.4f} ± {np.std(hand_list):.4f}\n")

if __name__ == '__main__':
    main()