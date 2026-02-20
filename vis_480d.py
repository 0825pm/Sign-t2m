"""
vis_480d.py — 480D 전처리 결과 검증 시각화

480D npy에서 joint positions (처음 120D) 추출 → skeleton 영상
FK 불필요! positions가 이미 pelvis-centered 3D 좌표.

Usage:
    python vis_480d.py --data_dir /path/to/Phoenix_480d --split train --num 5
    python vis_480d.py --npy /path/to/sample.npy              # 단일 파일
"""

import os, sys, argparse, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ═══════════════════════════════════════════════════════════
# 44-joint skeleton definition
# ═══════════════════════════════════════════════════════════
# Our 44 joints (from SMPLX output):
#   0:  Pelvis
#   1:  Spine1
#   2:  Spine2
#   3:  Spine3
#   4-13:  upper body [Neck, L_Collar, R_Collar, Head, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist]
#   14-28: left hand  [Index×3, Middle×3, Pinky×3, Ring×3, Thumb×3]
#   29-43: right hand  [Index×3, Middle×3, Pinky×3, Ring×3, Thumb×3]

N_JOINTS = 44
POS_DIM = N_JOINTS * 3      # 132
VEL_DIM = N_JOINTS * 3      # 132
ROT_DIM = N_JOINTS * 6      # 264
FEAT_DIM = POS_DIM + VEL_DIM + ROT_DIM  # 528

# Spine chain
SPINE_CONNECTIONS = [
    (0, 1),   # Pelvis → Spine1
    (1, 2),   # Spine1 → Spine2
    (2, 3),   # Spine2 → Spine3
    (3, 4),   # Spine3 → Neck
]

# Upper body (starting from idx 4)
BODY_CONNECTIONS = [
    (4, 7),   # Neck → Head
    (4, 5),   # Neck → L_Collar
    (4, 6),   # Neck → R_Collar
    (5, 8),   # L_Collar → L_Shoulder
    (6, 9),   # R_Collar → R_Shoulder
    (8, 10),  # L_Shoulder → L_Elbow
    (9, 11),  # R_Shoulder → R_Elbow
    (10, 12), # L_Elbow → L_Wrist
    (11, 13), # R_Elbow → R_Wrist
]

def _hand_connections(wrist_idx, hand_offset):
    conns = []
    for finger in range(5):
        base = hand_offset + finger * 3
        conns.append((wrist_idx, base))
        conns.append((base, base + 1))
        conns.append((base + 1, base + 2))
    return conns

LHAND_CONNECTIONS = _hand_connections(12, 14)   # L_Wrist=12, lhand starts at 14
RHAND_CONNECTIONS = _hand_connections(13, 29)   # R_Wrist=13, rhand starts at 29
ALL_CONNECTIONS = SPINE_CONNECTIONS + BODY_CONNECTIONS + LHAND_CONNECTIONS + RHAND_CONNECTIONS

BODY_INDICES = list(range(14))    # spine + upper body
LHAND_INDICES = list(range(14, 29))
RHAND_INDICES = list(range(29, 44))


# ═══════════════════════════════════════════════════════════
# 480D → 40 joints
# ═══════════════════════════════════════════════════════════

def extract_positions(motion):
    """[T, 528] → [T, 44, 3] joint positions"""
    return motion[:, :POS_DIM].reshape(-1, N_JOINTS, 3)

def extract_velocities(motion):
    """[T, 528] → [T, 44, 3] joint velocities"""
    return motion[:, POS_DIM:POS_DIM+VEL_DIM].reshape(-1, N_JOINTS, 3)

def extract_6d_rot(motion):
    """[T, 528] → [T, 44, 6] 6D rotations"""
    return motion[:, POS_DIM+VEL_DIM:].reshape(-1, N_JOINTS, 6)


# ═══════════════════════════════════════════════════════════
# Skeleton video
# ═══════════════════════════════════════════════════════════

def save_skeleton_video(joints_44, save_path, title='', fps=24, viewport=None):
    """
    Args:
        joints_44: [T, 44, 3] pelvis-centered joint positions
    """
    T = joints_44.shape[0]

    # Center at Spine3 (idx 3, just below neck)
    root = joints_44[:, 3:4, :]
    data = joints_44 - root

    # Auto viewport
    if viewport is None or viewport <= 0:
        all_pts = data.reshape(-1, 3)
        margin = 0.15
        x_range = all_pts[:, 0].max() - all_pts[:, 0].min()
        y_range = all_pts[:, 1].max() - all_pts[:, 1].min()
        vp = max(x_range, y_range) / 2 + margin
        vp = max(vp, 0.3)
    else:
        vp = viewport

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(title, fontsize=9, wrap=True)
    ax.set_xlim(-vp, vp)
    ax.set_ylim(-vp, vp)
    ax.set_aspect('equal')
    ax.axis('off')

    lines = []
    for (i, j) in ALL_CONNECTIONS:
        if i in LHAND_INDICES or j in LHAND_INDICES:
            c, lw = 'red', 0.8
        elif i in RHAND_INDICES or j in RHAND_INDICES:
            c, lw = 'green', 0.8
        elif (i, j) in SPINE_CONNECTIONS:
            c, lw = 'purple', 2.5
        else:
            c, lw = 'blue', 2.0
        line, = ax.plot([], [], color=c, linewidth=lw, alpha=0.8)
        lines.append((line, i, j))

    body_sc = ax.scatter([], [], c='blue', s=15, zorder=5)
    spine_sc = ax.scatter([], [], c='purple', s=20, zorder=5)
    lhand_sc = ax.scatter([], [], c='red', s=5, zorder=5)
    rhand_sc = ax.scatter([], [], c='green', s=5, zorder=5)
    frame_txt = fig.text(0.5, 0.02, '', ha='center', fontsize=9, color='gray')
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])

    def update(frame):
        f = min(frame, T - 1)
        frame_txt.set_text(f'Frame {f}/{T-1}')
        x, y = data[f, :, 0], -data[f, :, 1]
        for (line, i, j) in lines:
            line.set_data([x[i], x[j]], [y[i], y[j]])
        spine_sc.set_offsets(np.c_[x[:4], y[:4]])
        body_sc.set_offsets(np.c_[x[4:14], y[4:14]])
        lhand_sc.set_offsets(np.c_[x[14:29], y[14:29]])
        rhand_sc.set_offsets(np.c_[x[29:44], y[29:44]])
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception:
        anim.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=min(fps, 10))
    plt.close(fig)


def save_frame_image(joints_44, save_path, frame_idx=0, title=''):
    """Single frame snapshot"""
    data = joints_44 - joints_44[:, 3:4, :]  # center at Spine3
    f = min(frame_idx, len(data) - 1)
    x, y = data[f, :, 0], -data[f, :, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title, fontsize=9)

    for (i, j) in ALL_CONNECTIONS:
        if i in LHAND_INDICES or j in LHAND_INDICES:
            c, lw = 'red', 0.8
        elif i in RHAND_INDICES or j in RHAND_INDICES:
            c, lw = 'green', 0.8
        elif (i, j) in SPINE_CONNECTIONS:
            c, lw = 'purple', 2.5
        else:
            c, lw = 'blue', 2.0
        ax.plot([x[i], x[j]], [y[i], y[j]], color=c, linewidth=lw, alpha=0.8)

    ax.scatter(x[:4], y[:4], c='purple', s=25, zorder=5)
    ax.scatter(x[4:14], y[4:14], c='blue', s=20, zorder=5)
    ax.scatter(x[14:29], y[14:29], c='red', s=8, zorder=5)
    ax.scatter(x[29:44], y[29:44], c='green', s=8, zorder=5)

    # Label body joints
    names = ['Pelvis','Spine1','Spine2','Spine3',
             'Neck','L_Col','R_Col','Head','L_Shd','R_Shd','L_Elb','R_Elb','L_Wri','R_Wri']
    for idx in range(min(14, len(names))):
        ax.annotate(names[idx], (x[idx], y[idx]), fontsize=5,
                     ha='center', va='bottom', color='navy')

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Sanity checks
# ═══════════════════════════════════════════════════════════

def check_sample(motion_480, name=''):
    """Print diagnostic info for a single 480D sample"""
    T = motion_480.shape[0]
    pos = extract_positions(motion_480)    # [T, 40, 3]
    vel = extract_velocities(motion_480)   # [T, 40, 3]
    rot = extract_6d_rot(motion_480)       # [T, 40, 6]

    print(f"  {name}: T={T}, shape={motion_480.shape}")
    print(f"    positions:  range=[{pos.min():.4f}, {pos.max():.4f}], std={pos.std():.4f}")
    print(f"    velocities: range=[{vel.min():.4f}, {vel.max():.4f}], std={vel.std():.4f}")
    print(f"    6D_rot:     range=[{rot.min():.4f}, {rot.max():.4f}], std={rot.std():.4f}")

    # Velocity consistency check: vel[t] ≈ pos[t] - pos[t-1]
    if T > 1:
        computed_vel = np.diff(pos, axis=0)  # [T-1, 40, 3]
        stored_vel = vel[1:]                  # [T-1, 40, 3]
        vel_error = np.abs(computed_vel - stored_vel).mean()
        print(f"    vel consistency: mean_error={vel_error:.6f} {'✅' if vel_error < 0.001 else '⚠️'}")

    # 6D orthogonality check (first two columns should be orthonormal)
    r1 = rot[:, :, :3]  # first column
    r2 = rot[:, :, 3:]  # second column
    dot = np.sum(r1 * r2, axis=-1)
    norm1 = np.linalg.norm(r1, axis=-1)
    norm2 = np.linalg.norm(r2, axis=-1)
    print(f"    6D ortho:   dot={np.abs(dot).mean():.6f} (should≈0), "
          f"norm1={norm1.mean():.4f}, norm2={norm2.mean():.4f} (should≈1)")

    # NaN/Inf check
    n_nan = np.isnan(motion_480).sum()
    n_inf = np.isinf(motion_480).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"    ❌ NaN={n_nan}, Inf={n_inf}")
    else:
        print(f"    ✅ No NaN/Inf")

    return pos


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Visualize 480D preprocessed data')
    parser.add_argument('--npy', default=None, help='Single .npy file to visualize')
    parser.add_argument('--data_dir', default=None, help='Directory with .npy files')
    parser.add_argument('--split', default='train')
    parser.add_argument('--num', type=int, default=5, help='Number of samples')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--viewport', type=float, default=0, help='0=auto')
    parser.add_argument('--output', default='vis_480d_output')
    parser.add_argument('--video', action='store_true', help='Also save video (slower)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Collect .npy files
    if args.npy:
        npy_files = [args.npy]
    elif args.data_dir:
        # Try common structures
        candidates = [
            os.path.join(args.data_dir, args.split, 'poses'),
            os.path.join(args.data_dir, args.split),
            os.path.join(args.data_dir, 'poses'),
            args.data_dir,
        ]
        npy_dir = None
        for c in candidates:
            if os.path.exists(c) and any(f.endswith('.npy') for f in os.listdir(c)):
                npy_dir = c
                break
        if npy_dir is None:
            print(f"No .npy files found in {args.data_dir}")
            return
        npy_files = sorted(glob.glob(os.path.join(npy_dir, '*.npy')))
        print(f"Found {len(npy_files)} files in {npy_dir}")
    else:
        print("Specify --npy or --data_dir")
        return

    # Select samples
    n = min(args.num, len(npy_files))
    indices = np.linspace(0, len(npy_files) - 1, n, dtype=int)
    selected = [npy_files[i] for i in indices]

    print(f"\n{'='*60}")
    print(f"  480D Visualization — {n} samples")
    print(f"{'='*60}")

    for i, path in enumerate(selected):
        name = os.path.splitext(os.path.basename(path))[0]
        motion = np.load(path)  # [T, 480]

        if motion.shape[-1] != FEAT_DIM:
            print(f"\n[{i+1}/{n}] {name}: unexpected dim {motion.shape[-1]}, skip")
            continue

        print(f"\n[{i+1}/{n}] {name}")
        pos = check_sample(motion, name)

        # Frame snapshot (frame 0, middle, last)
        for fi, label in [(0, 'first'), (len(pos)//2, 'mid'), (len(pos)-1, 'last')]:
            img_path = os.path.join(args.output, f'{i:03d}_{name}_{label}.png')
            save_frame_image(pos, img_path, fi, f'{name} frame={fi}')

        # Video
        if args.video:
            vid_path = os.path.join(args.output, f'{i:03d}_{name}.mp4')
            save_skeleton_video(pos, vid_path, f'{name} (T={len(pos)})',
                                args.fps, args.viewport if args.viewport > 0 else None)
            print(f"    ✅ Video: {vid_path}")

    print(f"\n{'='*60}")
    print(f"  Done. Outputs in {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()