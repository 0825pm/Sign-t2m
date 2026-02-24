"""
check_360d_loader.py — 360D 포즈 시각화 (position 직접 사용, FK 불필요)

360D 레이아웃:
  [0:30]    body_pos  (10j × 3, pelvis-relative)
  [30:90]   body_rot  (10j × 6D)
  [90:135]  lhand_pos (15j × 3, L_Wrist-relative)
  [135:225] lhand_rot (15j × 6D)
  [225:270] rhand_pos (15j × 3, R_Wrist-relative)
  [270:360] rhand_rot (15j × 6D)

Usage:
    python check_360d_loader.py --dataset how2sign --num_samples 5
"""

import os, sys, argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE     = '/home/user/Projects/research/SOKE/data'
DATA360  = f'{BASE}/data360'

# ── skeleton 연결 (40-joint 기준) ─────────────────────────────
# body[0:10]: 0:Pelvis, 1:Sp3, 2:Neck, 3:Head,
#             4:LSh, 5:RSh, 6:LElb, 7:RElb, 8:LWr, 9:RWr
# lhand[10:25], rhand[25:40]

SPINE = [(0,1),(1,2),(2,3)]
UPPER = [(2,4),(2,5),(4,6),(5,7),(6,8),(7,9)]

def _hand_conn(wrist, base):
    c = []
    for f in range(5):
        b = base + f*3
        c += [(wrist,b),(b,b+1),(b+1,b+2)]
    return c

ALL_CONN = SPINE + UPPER + _hand_conn(8,10) + _hand_conn(9,25)

LWRIST = 8  # body 내 L_Wrist 인덱스
RWRIST = 9  # body 내 R_Wrist 인덱스


def extract_joints(motion360):
    """
    [T, 360] → [T, 40, 3] 절대 좌표
    position만 사용, rotation 무시
    """
    T = motion360.shape[0]
    body_pos  = motion360[:, 0:30].reshape(T, 10, 3)   # pelvis-relative
    lhand_pos = motion360[:, 90:135].reshape(T, 15, 3)  # LWrist-relative
    rhand_pos = motion360[:, 225:270].reshape(T, 15, 3) # RWrist-relative

    # 절대 좌표로 복원
    lhand_abs = lhand_pos + body_pos[:, LWRIST:LWRIST+1, :]
    rhand_abs = rhand_pos + body_pos[:, RWRIST:RWRIST+1, :]

    return np.concatenate([body_pos, lhand_abs, rhand_abs], axis=1)  # [T,40,3]


def save_video(joints40, save_path, title='', fps=25):
    T = joints40.shape[0]
    data = joints40  # 이미 pelvis=origin

    flat_x = data[:,:,0].ravel()
    flat_y = data[:,:,1].ravel()
    cx = (flat_x.max()+flat_x.min())/2
    cy = (flat_y.max()+flat_y.min())/2
    vp = max(flat_x.max()-flat_x.min(), flat_y.max()-flat_y.min())/2 + 0.1
    vp = max(vp, 0.3)

    fig, ax = plt.subplots(figsize=(5,6))
    fig.suptitle(title[:80], fontsize=7)
    ax.set_xlim(cx-vp, cx+vp); ax.set_ylim(cy-vp, cy+vp)
    ax.set_aspect('equal'); ax.axis('off')

    lines = []
    for (i,j) in ALL_CONN:
        if i>=25 or j>=25:      c,lw = '#4CAF50', 0.9   # rhand
        elif i>=10 or j>=10:    c,lw = '#E91E63', 0.9   # lhand
        elif (i,j) in SPINE:    c,lw = 'purple',  2.5
        else:                   c,lw = '#333333',  2.0
        ln, = ax.plot([],[],color=c,linewidth=lw,alpha=0.85)
        lines.append((ln,i,j))
    sc_b = ax.scatter([],[],c='#333333',s=25,zorder=5)
    sc_l = ax.scatter([],[],c='#E91E63',s=8,zorder=5)
    sc_r = ax.scatter([],[],c='#4CAF50',s=8,zorder=5)
    ftxt = fig.text(0.5,0.01,'',ha='center',fontsize=7,color='gray')
    plt.tight_layout(rect=[0,0.03,1,0.93])

    def update(f):
        x=data[f,:,0]; y=data[f,:,1]
        for (ln,i,j) in lines:
            ln.set_data([x[i],x[j]],[y[i],y[j]])
        sc_b.set_offsets(np.c_[x[:10],  y[:10]])
        sc_l.set_offsets(np.c_[x[10:25],y[10:25]])
        sc_r.set_offsets(np.c_[x[25:40],y[25:40]])
        ftxt.set_text(f'Frame {f}/{T-1}')
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    try:
        anim.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception:
        anim.save(save_path.replace('.mp4','.gif'), writer='pillow', fps=min(fps,10))
    plt.close(fig)
    print(f"  → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='how2sign',
                        choices=['how2sign','csl','phoenix'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--output', default='check_360d_output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # data360 경로
    if args.dataset == 'how2sign':
        data_root = f'{DATA360}/How2Sign'
    elif args.dataset == 'csl':
        data_root = f'{DATA360}/CSL-Daily'
    else:
        data_root = f'{DATA360}/Phoenix_2014T'

    mean_path = os.path.join(data_root, 'mean_360.pt')
    std_path  = os.path.join(data_root, 'std_360.pt')

    mean = torch.load(mean_path, map_location='cpu').float()
    std  = torch.load(std_path,  map_location='cpu').float()
    mean_np, std_np = mean.numpy(), std.numpy()

    # annotation은 원본, npy는 data360
    from src.data.signlang.dataset_sign import SignMotionDataset
    ds = SignMotionDataset(
        data_root=f'{BASE}/How2Sign',
        csl_root=f'{BASE}/CSL-Daily',
        phoenix_root=f'{BASE}/Phoenix_2014T',
        npy_root=f'{DATA360}/How2Sign',
        csl_npy_root=f'{DATA360}/CSL-Daily',
        phoenix_npy_root=f'{DATA360}/Phoenix_2014T',
        split=args.split, mean=mean, std=std,
        nfeats=360, dataset_name=args.dataset,
        max_motion_length=10000, min_motion_length=1,
    )
    print(f"Dataset: {len(ds)} samples\n")

    indices = np.linspace(0, len(ds)-1, args.num_samples, dtype=int)
    for rank, idx in enumerate(indices):
        item = ds[int(idx)]
        T_len = int(item['motion_len'])
        text  = item['text']
        name  = item['name']
        src   = item['src']

        # 역정규화
        raw360 = item['motion'].numpy()[:T_len] * (std_np+1e-10) + mean_np

        print(f"[{rank+1}/{args.num_samples}] {name[:50]}  T={T_len}")
        print(f"  text: \"{text[:65]}\"")

        joints40 = extract_joints(raw360)

        save_video(
            joints40,
            os.path.join(args.output, f'{rank:03d}_{name.replace("/","_")[:40]}.mp4'),
            title=f'[{src}] {text[:55]}\nT={T_len}',
            fps=args.fps
        )
        print()


if __name__ == '__main__':
    main()