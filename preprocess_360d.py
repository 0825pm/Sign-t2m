"""
preprocess_360d.py — 133D axis-angle → 360D (position + 6D rotation) npy

SignMotionDataset(nfeats=133)으로 로드 후 FK → 360D 변환.
annotation 경로 탐색 불필요.

360D 레이아웃:
  [0:30]    body_pos  (10j × 3, pelvis-relative)
  [30:90]   body_rot  (10j × 6D)
  [90:135]  lhand_pos (15j × 3, L_Wrist-relative)
  [135:225] lhand_rot (15j × 6D)
  [225:270] rhand_pos (15j × 3, R_Wrist-relative)
  [270:360] rhand_rot (15j × 6D)

Usage:
    python preprocess_360d.py --dataset how2sign
    python preprocess_360d.py --dataset all
"""

import os, sys, argparse
import numpy as np
import torch
from tqdm import tqdm

torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE     = '/home/user/Projects/research/SOKE/data'
OUT_BASE = f'{BASE}/data360'

# check_133d_loader.py와 동일한 SMPLX joint 인덱스
BODY_IDX  = [0, 9, 12, 15, 16, 17, 18, 19, 20, 21]
LHAND_IDX = list(range(25, 40))
RHAND_IDX = list(range(40, 55))
SEL       = BODY_IDX + LHAND_IDX + RHAND_IDX  # 40 joints

LWRIST = 8   # SEL 내 L_Wrist 인덱스 (body[8])
RWRIST = 9   # SEL 내 R_Wrist 인덱스 (body[9])


def aa_to_6d(aa_np):
    """axis-angle [T, N, 3] → 6D [T, N, 6] (numpy)"""
    T, N, _ = aa_np.shape
    aa_flat = aa_np.reshape(-1, 3)
    angle = np.linalg.norm(aa_flat, axis=-1, keepdims=True).clip(1e-8)
    axis  = aa_flat / angle
    c = np.cos(angle); s = np.sin(angle); t_c = 1 - c
    x, y, z = axis[:,0:1], axis[:,1:2], axis[:,2:3]
    R = np.stack([
        t_c*x*x+c,   t_c*x*y-s*z, t_c*x*z+s*y,
        t_c*x*y+s*z, t_c*y*y+c,   t_c*y*z-s*x,
        t_c*x*z-s*y, t_c*y*z+s*x, t_c*z*z+c,
    ], axis=-1).reshape(-1, 3, 3)
    rot6d = np.concatenate([R[:,:,0], R[:,:,1]], axis=-1)  # [N,6]
    return rot6d.reshape(T, N, 6).astype(np.float32)


def raw133_to_360(raw133):
    """
    [T, 133] axis-angle → [T, 360] position+6D rotation

    133D 레이아웃:
      [0:30]   upper_body aa (10j×3)
      [30:75]  lhand aa (15j×3)
      [75:120] rhand aa (15j×3)
      [120:123] jaw (무시)
      [123:133] expr (무시)
    """
    from src.utils.human_models import get_coord

    T = raw133.shape[0]
    upper_body_aa = raw133[:, 0:30].reshape(T, 10, 3)
    lhand_aa      = raw133[:, 30:75]
    rhand_aa      = raw133[:, 75:120]

    # SMPLX FK → positions
    lower_zeros  = np.zeros((T, 33), dtype=np.float32)
    body_pose_np = np.concatenate([lower_zeros, raw133[:, 0:30]], axis=-1)

    def t(x): return torch.from_numpy(x).float().cuda()
    shape = t(np.tile(
        np.array([-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
                   0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842],
                 dtype=np.float32), (T, 1)))

    _, joints_all = get_coord(
        t(np.zeros((T,3),  dtype=np.float32)),
        t(body_pose_np),
        t(lhand_aa), t(rhand_aa),
        t(np.zeros((T,3),  dtype=np.float32)),
        shape,
        t(np.zeros((T,10), dtype=np.float32)),
    )
    joints40 = joints_all.cpu().numpy()[:, SEL, :]  # [T,40,3]

    # Position (relative)
    body_pos  = joints40[:, :10, :] - joints40[:, 0:1, :]           # pelvis-relative
    lhand_pos = joints40[:, 10:25, :] - joints40[:, LWRIST:LWRIST+1, :]  # wrist-relative
    rhand_pos = joints40[:, 25:40, :] - joints40[:, RWRIST:RWRIST+1, :]

    # Rotation (6D)
    body_rot  = aa_to_6d(upper_body_aa)                              # [T,10,6]
    lhand_rot = aa_to_6d(lhand_aa.reshape(T, 15, 3))                 # [T,15,6]
    rhand_rot = aa_to_6d(rhand_aa.reshape(T, 15, 3))                 # [T,15,6]

    return np.concatenate([
        body_pos.reshape(T, 30),   # [0:30]
        body_rot.reshape(T, 60),   # [30:90]
        lhand_pos.reshape(T, 45),  # [90:135]
        lhand_rot.reshape(T, 90),  # [135:225]
        rhand_pos.reshape(T, 45),  # [225:270]
        rhand_rot.reshape(T, 90),  # [270:360]
    ], axis=-1).astype(np.float32)


def process_split(dataset_name, split, mean_np, std_np):
    """SignMotionDataset으로 로드 → 360D npy 저장"""
    from src.data.signlang.dataset_sign import SignMotionDataset

    if dataset_name == 'how2sign':
        out_root = f'{OUT_BASE}/How2Sign'
    elif dataset_name == 'csl':
        out_root = f'{OUT_BASE}/CSL-Daily'
    else:
        out_root = f'{OUT_BASE}/Phoenix_2014T'

    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    mean = torch.from_numpy(mean_np).float()
    std  = torch.from_numpy(std_np).float()

    ds = SignMotionDataset(
        data_root=f'{BASE}/How2Sign',
        csl_root=f'{BASE}/CSL-Daily',
        phoenix_root=f'{BASE}/Phoenix_2014T',
        split=split,
        mean=mean, std=std,
        nfeats=133,
        dataset_name=dataset_name,
        max_motion_length=10000,
        min_motion_length=1,
    )
    print(f"  [{dataset_name}/{split}] {len(ds)} samples")

    all_motions = []
    for i in tqdm(range(len(ds)), desc=f'{dataset_name}/{split}'):
        item = ds[i]
        name = item['name']
        motion_norm = item['motion'].numpy()  # [T, 133] normalized
        T_len = int(item['motion_len'])

        # 역정규화
        raw133 = motion_norm[:T_len] * (std_np + 1e-10) + mean_np

        save_name = name.replace('/', '_')
        out_path  = os.path.join(out_dir, f'{save_name}.npy')
        if os.path.exists(out_path):
            if split == 'train':
                all_motions.append(np.load(out_path))
            continue

        try:
            motion360 = raw133_to_360(raw133)
            np.save(out_path, motion360)
            if split == 'train':
                all_motions.append(motion360)
        except Exception as e:
            print(f"\n  ⚠ {name}: {e}")

    return all_motions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='how2sign',
                        choices=['how2sign', 'csl', 'phoenix', 'all'])
    args = parser.parse_args()

    datasets = ['how2sign', 'csl', 'phoenix'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        print(f"\n{'='*50}\nProcessing {ds} ...")

        # mean/std (133D)
        mean_np = torch.load(f'{BASE}/How2Sign/mean_133.pt', map_location='cpu').numpy()
        std_np  = torch.load(f'{BASE}/How2Sign/std_133.pt',  map_location='cpu').numpy()

        all_train_motions = []
        for split in ['train', 'val', 'test']:
            motions = process_split(ds, split, mean_np, std_np)
            all_train_motions.extend(motions)

        # mean/std 360D 계산 (train 기반)
        if all_train_motions:
            out_root = f'{OUT_BASE}/How2Sign' if ds == 'how2sign' else \
                       f'{OUT_BASE}/CSL-Daily' if ds == 'csl' else \
                       f'{OUT_BASE}/Phoenix_2014T'
            cat  = np.concatenate(all_train_motions, axis=0)
            mean = cat.mean(axis=0).astype(np.float32)
            std  = cat.std(axis=0).clip(1e-6, None).astype(np.float32)
            torch.save(torch.from_numpy(mean), f'{out_root}/mean_360.pt')
            torch.save(torch.from_numpy(std),  f'{out_root}/std_360.pt')
            print(f"\n  mean range=[{mean.min():.4f},{mean.max():.4f}]  "
                  f"std range=[{std.min():.4f},{std.max():.4f}]")
            print(f"  저장: {out_root}/mean_360.pt, std_360.pt")

    print("\n✅ 완료")


if __name__ == '__main__':
    main()