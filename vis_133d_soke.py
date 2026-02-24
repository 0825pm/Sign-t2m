"""
vis_133d_soke.py — 133D npy 데이터 로딩 검증 시각화

기존 vis_generation.py와 동일한 방식으로 SignText2MotionDataset을 직접 사용.

Usage:
    cd ~/Projects/research/sign-t2m

    python vis_133d_soke.py \
        --mean_path /home/user/Projects/research/SOKE/data/How2Sign_133d/mean_133.pt \
        --std_path  /home/user/Projects/research/SOKE/data/How2Sign_133d/std_133.pt \
        --npy_root  /home/user/Projects/research/SOKE/data/How2Sign_133d \
        --num_samples 5

    # CSL-Daily
    python vis_133d_soke.py \
        --dataset csl \
        --csl_npy_root /home/user/Projects/research/SOKE/data/CSL-Daily_133d \
        --csl_mean_path /home/user/Projects/research/SOKE/data/CSL-Daily_133d/mean_133.pt \
        --csl_std_path  /home/user/Projects/research/SOKE/data/CSL-Daily_133d/std_133.pt
"""

import os
import sys
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

torch.backends.cudnn.enabled = False   # Mamba cuDNN 충돌 방지
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 44-joint skeleton (vis_generation.py 동일)
# =============================================================================
SPINE_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4)]
BODY_CONNECTIONS  = [(4,7),(4,5),(4,6),(5,8),(6,9),(8,10),(9,11),(10,12),(11,13)]

def _hand_conns(wrist, offset):
    c = []
    for f in range(5):
        b = offset + f * 3
        c += [(wrist, b), (b, b+1), (b+1, b+2)]
    return c

LHAND_CONNECTIONS = _hand_conns(12, 14)
RHAND_CONNECTIONS = _hand_conns(13, 29)
ALL_CONNECTIONS   = SPINE_CONNECTIONS + BODY_CONNECTIONS + LHAND_CONNECTIONS + RHAND_CONNECTIONS
LHAND_INDICES     = list(range(14, 29))
RHAND_INDICES     = list(range(29, 44))


# =============================================================================
# 133D → 44-joint 좌표 복원
# FINDINGS.md 기준:
#   [0:4]    root_motion
#   [4:43]   body_ric (13 joints × 3, pelvis-relative)
#   [43:88]  lhand_ric (15 joints × 3, wrist-relative)
#   [88:133] rhand_ric (15 joints × 3, wrist-relative)
# =============================================================================
def feats133_to_joints44(feat_133):
    """역정규화된 133D → [T, 44, 3]"""
    T = feat_133.shape[0]
    joints = np.zeros((T, 44, 3), dtype=np.float32)

    body_ric  = feat_133[:, 4:43].reshape(T, 13, 3)
    lhand_ric = feat_133[:, 43:88].reshape(T, 15, 3)
    rhand_ric = feat_133[:, 88:133].reshape(T, 15, 3)

    joints[:, 0, :]    = 0.0        # pelvis
    joints[:, 1:14, :] = body_ric   # body joints 1-13

    lwrist = joints[:, 12:13, :]    # body joint 12 = lwrist
    rwrist = joints[:, 13:14, :]    # body joint 13 = rwrist

    joints[:, 14:29, :] = lhand_ric + lwrist   # wrist-relative → absolute
    joints[:, 29:44, :] = rhand_ric + rwrist

    return joints


# =============================================================================
# Skeleton 렌더링 (vis_generation.py 동일 구조)
# =============================================================================
def _setup_ax(ax, label, color, xlim, ylim):
    ax.set_title(label, fontsize=11, fontweight='bold', color=color)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal'); ax.axis('off')


def _get_viewport(data, viewport=0):
    if viewport > 0:
        return (-viewport, viewport), (-viewport, viewport)
    pts = data.reshape(-1, 3)
    margin = 0.1
    vp = max(
        pts[:, 0].max() - pts[:, 0].min(),
        pts[:, 1].max() - pts[:, 1].min()
    ) / 2 + margin
    return (-max(vp, 0.2), max(vp, 0.2)), (-max(vp, 0.2), max(vp, 0.2))


def _build_elements(ax, colors):
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
    lhand_sc = ax.scatter([], [], c=colors['lhand'], s=6,  zorder=5)
    rhand_sc = ax.scatter([], [], c=colors['rhand'], s=6,  zorder=5)
    return lines, spine_sc, body_sc, lhand_sc, rhand_sc


def _update(data, f, lines, spine_sc, body_sc, lhand_sc, rhand_sc):
    x, y = data[f, :, 0], -data[f, :, 1]
    for (line, i, j) in lines:
        line.set_data([x[i], x[j]], [y[i], y[j]])
    spine_sc.set_offsets(np.c_[x[:4],    y[:4]])
    body_sc .set_offsets(np.c_[x[4:14],  y[4:14]])
    lhand_sc.set_offsets(np.c_[x[14:29], y[14:29]])
    rhand_sc.set_offsets(np.c_[x[29:44], y[29:44]])


def save_single_video(joints, save_path, title='', fps=25, viewport=0):
    T = joints.shape[0]
    data = joints - joints[:, 3:4, :]   # spine3 중심
    xl, yl = _get_viewport(data, viewport)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(title, fontsize=9)
    _setup_ax(ax, '', 'blue', xl, yl)
    el = _build_elements(ax, {'body':'blue', 'lhand':'#E91E63', 'rhand':'#4CAF50'})
    ftxt = fig.text(0.5, 0.02, '', ha='center', fontsize=8, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    def update(frame):
        f = min(frame, T-1)
        ftxt.set_text(f'Frame {f}/{T-1}')
        _update(data, f, *el)
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=5000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='133D npy 데이터 로딩 검증 시각화')
    parser.add_argument('--data_root',
        default='/home/user/Projects/research/SOKE/data/How2Sign')
    parser.add_argument('--csl_root',
        default='/home/user/Projects/research/SOKE/data/CSL-Daily')
    parser.add_argument('--phoenix_root',
        default='/home/user/Projects/research/SOKE/data/Phoenix_2014T')
    parser.add_argument('--npy_root',
        default='/home/user/Projects/research/SOKE/data/How2Sign_133d')
    parser.add_argument('--csl_npy_root',
        default='/home/user/Projects/research/SOKE/data/CSL-Daily_133d')
    parser.add_argument('--phoenix_npy_root',
        default='/home/user/Projects/research/SOKE/data/Phoenix_133d')
    parser.add_argument('--mean_path',
        default='/home/user/Projects/research/SOKE/data/How2Sign_133d/mean_133.pt')
    parser.add_argument('--std_path',
        default='/home/user/Projects/research/SOKE/data/How2Sign_133d/std_133.pt')
    parser.add_argument('--csl_mean_path',
        default='/home/user/Projects/research/SOKE/data/CSL-Daily_133d/mean_133.pt')
    parser.add_argument('--csl_std_path',
        default='/home/user/Projects/research/SOKE/data/CSL-Daily_133d/std_133.pt')
    parser.add_argument('--dataset', default='how2sign')
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0)
    parser.add_argument('--output', default='vis_133d_output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── mean/std 로드 ──────────────────────────────────────
    mean = torch.load(args.mean_path, map_location='cpu').float()
    std  = torch.load(args.std_path,  map_location='cpu').float()

    csl_mean = csl_std = None
    if os.path.exists(args.csl_mean_path):
        csl_mean = torch.load(args.csl_mean_path, map_location='cpu').float()
        csl_std  = torch.load(args.csl_std_path,  map_location='cpu').float()

    print(f"mean: {mean.shape}, [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"std:  {std.shape},  [{std.min():.4f}, {std.max():.4f}]")

    # ── SignText2MotionDataset 로드 (dataset_sign.py 그대로) ─
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
        nfeats=133,
        dataset_name=args.dataset,
        max_motion_length=400,
        min_motion_length=20,
    )
    print(f"Dataset: {len(dataset)} samples")

    # ── 균등 샘플링 + 시각화 ───────────────────────────────
    indices = np.linspace(0, len(dataset)-1, args.num_samples, dtype=int)

    for rank, idx in enumerate(indices):
        item = dataset[idx]

        motion_norm = item['motion'].numpy()   # [T, 133] normalized
        text  = item['text']
        name  = item['name']
        src   = item['src']
        T_len = item['motion_len']

        # 역정규화
        if src == 'csl' and csl_mean is not None:
            m_np = csl_mean.numpy()
            s_np = csl_std.numpy()
        else:
            m_np = mean.numpy()
            s_np = std.numpy()

        raw = motion_norm * (s_np + 1e-10) + m_np   # [T, 133]

        # 간단 통계
        body_std  = raw[:, 4:43].std()
        lhand_std = raw[:, 43:88].std()
        rhand_std = raw[:, 88:133].std()
        print(f"[{rank+1}] {name[:45]}  T={T_len}  src={src}")
        print(f"     body={body_std:.4f}  lhand={lhand_std:.4f}  rhand={rhand_std:.4f}")

        # 44-joint 복원 후 저장
        joints = feats133_to_joints44(raw)
        safe_name = name.replace('/', '_')[:50]
        save_path = os.path.join(args.output, f'{rank:03d}_{safe_name}.mp4')
        save_single_video(joints, save_path,
                          title=f'[{src}] {text[:60]}\nT={T_len}',
                          fps=args.fps, viewport=args.viewport)
        print(f"     → {save_path}")

    print(f"\n완료. output: {os.path.abspath(args.output)}")
    print("[체크] lhand_std > 0, rhand_std > 0 이어야 데이터 정상")


if __name__ == '__main__':
    main()