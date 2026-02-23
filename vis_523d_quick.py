"""
vis_523d_quick.py — 523D npy 데이터 빠른 검증 시각화

523D layout:
  Body [0:163]:   root_motion(4) + body_ric(39) + body_rot(78) + body_vel(42)
  LHand [163:343]: lhand_ric(45) + lhand_rot(90) + lhand_vel(45) [wrist-relative]
  RHand [343:523]: rhand_ric(45) + rhand_rot(90) + rhand_vel(45) [wrist-relative]

복원: body_ric → pelvis-relative 14 joints, hand_ric + wrist → 44 joints

Usage:
    python vis_523d_quick.py
    python vis_523d_quick.py --dataset how2sign --num 5
    python vis_523d_quick.py --dataset phoenix --num 3
    python vis_523d_quick.py --npy_path /path/to/specific.npy
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =============================================================================
# 523D → 44-joint positions
# =============================================================================

def npy523_to_joints(data):
    """
    [T, 523] → [T, 44, 3]

    body_ric[4:43] = 13 joints pelvis-relative (pelvis=원점)
    lhand_ric[163:208] = 15 joints lwrist-relative
    rhand_ric[343:388] = 15 joints rwrist-relative
    """
    T = data.shape[0]

    # Body: pelvis(원점) + 13 joints
    pelvis = np.zeros((T, 1, 3), dtype=np.float32)
    body_ric = data[:, 4:43].reshape(T, 13, 3)
    body_14 = np.concatenate([pelvis, body_ric], axis=1)  # [T, 14, 3]

    # Wrist positions (body idx 12=lwrist, 13=rwrist)
    lwrist = body_14[:, 12:13, :]  # [T, 1, 3]
    rwrist = body_14[:, 13:14, :]  # [T, 1, 3]

    # Hands: wrist-relative → pelvis-relative
    lhand = data[:, 163:208].reshape(T, 15, 3) + lwrist
    rhand = data[:, 343:388].reshape(T, 15, 3) + rwrist

    joints = np.concatenate([body_14, lhand, rhand], axis=1)  # [T, 44, 3]
    return joints


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


def render_video(joints, save_path, title='', fps=25, viewport=0):
    """joints: [T, 44, 3] → mp4 (vis_528d_quick.py 그대로)"""
    T = joints.shape[0]
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(cx - vp, cx + vp)
    ax.set_ylim(cy - vp, cy + vp)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9)

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
    frame_txt = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    def update(f):
        x = centered[f, :, 0]
        y = -centered[f, :, 1]
        for (line, i, j) in lines:
            line.set_data([x[i], x[j]], [y[i], y[j]])
        body_sc.set_offsets(np.c_[x[:14], y[:14]])
        lhand_sc.set_offsets(np.c_[x[14:29], y[14:29]])
        rhand_sc.set_offsets(np.c_[x[29:44], y[29:44]])
        frame_txt.set_text(f'Frame {f}/{T-1}')
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception:
        gif_path = save_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=min(fps, 10))
        save_path = gif_path
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='how2sign', choices=['how2sign', 'csl', 'phoenix'])
    parser.add_argument('--npy_path', default=None)
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument('--output', default='vis_523d_check')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0)
    args = parser.parse_args()

    BASE = '/home/user/Projects/research/SOKE/data'
    npy_dirs = {
        'how2sign': f'{BASE}/How2Sign_523d/train/poses',
        'csl':      f'{BASE}/CSL-Daily_523d/poses',
        'phoenix':  f'{BASE}/Phoenix_523d/train',
    }

    os.makedirs(args.output, exist_ok=True)

    if args.npy_path:
        npy_files = [args.npy_path]
    else:
        npy_dir = npy_dirs[args.dataset]
        all_files = sorted([os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')])
        idx = np.linspace(0, len(all_files)-1, min(args.num, len(all_files)), dtype=int)
        npy_files = [all_files[i] for i in idx]

    for i, npy_path in enumerate(npy_files):
        fname = os.path.basename(npy_path).replace('.npy', '')
        print(f'[{i+1}/{len(npy_files)}] {fname}')

        data = np.load(npy_path)
        print(f'  shape: {data.shape}')
        assert data.shape[1] == 523, f"Expected 523D, got {data.shape[1]}"

        joints = npy523_to_joints(data)
        diff = np.abs(np.diff(joints, axis=0)).mean()
        print(f'  frames={joints.shape[0]}, mean_diff={diff:.6f}m')

        # Hand stats (should be small range since wrist-relative)
        lhand_ric = data[:, 163:208].reshape(-1, 15, 3)
        rhand_ric = data[:, 343:388].reshape(-1, 15, 3)
        print(f'  lhand_ric range: [{lhand_ric.min():.4f}, {lhand_ric.max():.4f}]')
        print(f'  rhand_ric range: [{rhand_ric.min():.4f}, {rhand_ric.max():.4f}]')

        save_path = os.path.join(args.output, f'{i:02d}_{fname}.mp4')
        actual = render_video(joints, save_path,
                              title=f'{fname} (T={joints.shape[0]}) [523D]',
                              fps=args.fps, viewport=args.viewport)
        print(f'  → {actual}')

    print(f'\nDone! Check {args.output}/')


if __name__ == '__main__':
    main()