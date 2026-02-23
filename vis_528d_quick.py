"""
vis_528d_quick.py — 528D npy 데이터 빠른 검증 시각화

npy 파일은 raw (unnormalized) 상태로 저장됨 → denorm 불필요

Usage:
    python vis_528d_quick.py
    python vis_528d_quick.py --stats_dir /path/to/Phoenix_528d --num 5
    python vis_528d_quick.py --npy_path /path/to/specific.npy
    python vis_528d_quick.py --viewport 0.5          # 고정 viewport
    python vis_528d_quick.py --viewport 0             # auto viewport (기본)
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =============================================================================
# 44-joint skeleton
# =============================================================================
# 0-3: Pelvis, Spine1, Spine2, Spine3
# 4-13: Neck, L_Collar, R_Collar, Head, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist
# 14-28: left hand (15 joints, 5 fingers × 3)
# 29-43: right hand (15 joints, 5 fingers × 3)

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
    """
    joints: [T, 44, 3] → mp4/gif
    viewport: 0=auto, >0=fixed half-range (e.g. 0.5 → [-0.5, 0.5])
    """
    T = joints.shape[0]
    
    # Center at spine3 (idx 3)
    centered = joints - joints[:, 3:4, :]
    x_all = centered[:, :, 0]
    y_all = -centered[:, :, 1]  # flip Y for display
    
    if viewport > 0:
        cx, cy = 0, 0
        vp = viewport
    else:
        margin = 0.05
        xmin, xmax = x_all.min() - margin, x_all.max() + margin
        ymin, ymax = y_all.min() - margin, y_all.max() + margin
        vp = max(xmax - xmin, ymax - ymin) / 2
        vp = max(vp, 0.15)  # minimum viewport
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(cx - vp, cx + vp)
    ax.set_ylim(cy - vp, cy + vp)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9)
    
    # Build line objects
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
    parser.add_argument('--stats_dir', default='/home/user/Projects/research/SOKE/data/Phoenix_528d')
    parser.add_argument('--npy_path', default=None, help='specific npy file')
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument('--output', default='vis_528d_check')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--viewport', type=float, default=0,
                        help='0=auto, >0=fixed half-range (e.g. 0.5)')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Collect npy files
    if args.npy_path:
        npy_files = [args.npy_path]
    else:
        train_dir = os.path.join(args.stats_dir, 'train')
        all_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.npy')])
        idx = np.linspace(0, len(all_files)-1, min(args.num, len(all_files)), dtype=int)
        npy_files = [all_files[i] for i in idx]
    
    for i, npy_path in enumerate(npy_files):
        fname = os.path.basename(npy_path).replace('.npy', '')
        print(f'[{i+1}/{len(npy_files)}] {fname}')
        
        # npy는 raw (unnormalized) → 바로 positions 추출
        data = np.load(npy_path)
        joints = data[:, :132].reshape(-1, 44, 3)
        
        diff = np.abs(np.diff(joints, axis=0)).mean()
        print(f'  frames={joints.shape[0]}, mean_diff={diff:.6f}m')
        
        save_path = os.path.join(args.output, f'{i:02d}_{fname}.mp4')
        actual = render_video(joints, save_path,
                              title=f'{fname} (T={joints.shape[0]})',
                              fps=args.fps, viewport=args.viewport)
        print(f'  → {actual}')
    
    print(f'\nDone! Check {args.output}/')


if __name__ == '__main__':
    main()