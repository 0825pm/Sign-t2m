"""
check_133d_loader.py — 133D 포즈 시각화 (상반신 + 양손만)
Usage:
    python check_133d_loader.py --dataset how2sign --num_samples 5
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = '/home/user/Projects/research/SOKE/data'

# ── SMPLX output.joints (144개) 인덱스 ───────────────────────
# 0-21:  body, 22:Jaw, 23-24:eyes, 25-39:lhand(15), 40-54:rhand(15), 55+:extra/face
BODY_IDX  = [0, 9, 12, 15, 16, 17, 18, 19, 20, 21]  # 상반신: Pelvis,Sp3,Neck,Head,LSh,RSh,LElb,RElb,LWr,RWr
LHAND_IDX = list(range(25, 40))   # 15개, jaw/eye 완전 제외
RHAND_IDX = list(range(40, 55))   # 15개
SEL = BODY_IDX + LHAND_IDX + RHAND_IDX  # 40 joints total

# ── 40-joint 내 skeleton 연결 ─────────────────────────────────
# body[0:10]: 0:Pelvis, 1:Sp3, 2:Neck, 3:Head, 4:LSh, 5:RSh, 6:LElb, 7:RElb, 8:LWr, 9:RWr
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


def joints_from_133d(raw133):
    """133D → [T,40,3] 상반신+양손 joints via SMPLX FK"""
    from src.utils.human_models import get_coord
    T = raw133.shape[0]

    upper_body = raw133[:, 0:30]   # [T,30]
    lhand      = raw133[:, 30:75]  # [T,45]
    rhand      = raw133[:, 75:120] # [T,45]
    # jaw/expr 제외 → zeros
    jaw_zeros  = np.zeros((T, 3),  dtype=np.float32)
    expr_zeros = np.zeros((T, 10), dtype=np.float32)

    # body_pose: 63D = lower(33) zeros + upper(30)
    body_pose_np = np.concatenate([np.zeros((T,33),dtype=np.float32), upper_body], axis=-1)

    def t(x): return torch.from_numpy(x).float().cuda()
    shape = t(np.tile(
        np.array([-0.07284723,0.1795129,-0.27608207,0.135155,0.10748172,
                   0.16037364,-0.01616933,-0.03450319,0.01369138,0.01108842],
                 dtype=np.float32), (T,1)))

    _, joints_all = get_coord(
        t(np.zeros((T,3), dtype=np.float32)),  # root
        t(body_pose_np),
        t(lhand), t(rhand),
        t(jaw_zeros), shape, t(expr_zeros)
    )
    joints_all = joints_all.cpu().numpy()  # [T, N, 3]
    assert joints_all.shape[1] >= 55, f"joint 수 부족: {joints_all.shape[1]}"
    return joints_all[:, SEL, :]   # [T, 40, 3]


def save_video(joints40, save_path, title='', fps=25):
    T = joints40.shape[0]
    data = joints40 - joints40[:, 0:1, :]   # Pelvis 중심

    flat_x = data[:,:,0].ravel()
    flat_y = data[:,:,1].ravel()
    cx = (flat_x.max()+flat_x.min())/2
    cy = (flat_y.max()+flat_y.min())/2
    vp = max(flat_x.max()-flat_x.min(), flat_y.max()-flat_y.min())/2 + 0.15
    vp = max(vp, 0.3)

    fig, ax = plt.subplots(figsize=(5,6))
    fig.suptitle(title[:80], fontsize=7)
    ax.set_xlim(cx-vp, cx+vp); ax.set_ylim(cy-vp, cy+vp)
    ax.set_aspect('equal'); ax.axis('off')

    lines = []
    for (i,j) in ALL_CONN:
        if i>=25 or j>=25:       c,lw = '#4CAF50', 0.9   # rhand
        elif i>=10 or j>=10:     c,lw = '#E91E63', 0.9   # lhand
        elif (i,j) in SPINE:     c,lw = 'purple',  2.5
        else:                    c,lw = '#333333',  2.0
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
    parser.add_argument('--dataset', default='how2sign', choices=['how2sign','csl','phoenix'])
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--output', default='check_133d_output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    mean = torch.load(f'{BASE}/How2Sign/mean_133.pt', map_location='cpu').float()
    std  = torch.load(f'{BASE}/How2Sign/std_133.pt',  map_location='cpu').float()
    mean_np, std_np = mean.numpy(), std.numpy()

    from src.data.signlang.dataset_sign import SignMotionDataset
    ds = SignMotionDataset(
        data_root=f'{BASE}/How2Sign', csl_root=f'{BASE}/CSL-Daily',
        phoenix_root=f'{BASE}/Phoenix_2014T',
        split=args.split, mean=mean, std=std,
        nfeats=133, dataset_name=args.dataset,
        max_motion_length=400, min_motion_length=20,
    )
    print(f"Dataset: {len(ds)} samples\n")

    indices = np.linspace(0, len(ds)-1, args.num_samples, dtype=int)
    for rank, idx in enumerate(indices):
        item = ds[int(idx)]
        raw = item['motion'].numpy() * (std_np+1e-10) + mean_np
        T_len = int(item['motion_len'])
        text  = item['text']; name = item['name']; src = item['src']
        print(f"[{rank+1}/{args.num_samples}] {name[:50]}  T={T_len}")
        print(f"  text: \"{text[:65]}\"")
        try:
            joints40 = joints_from_133d(raw)
            save_video(joints40,
                       os.path.join(args.output, f'{rank:03d}_{name.replace("/","_")[:40]}.mp4'),
                       title=f'[{src}] {text[:55]}\nT={T_len}', fps=args.fps)
        except Exception as e:
            print(f"  ⚠ FK 실패: {e}")
        print()

if __name__ == '__main__':
    main()