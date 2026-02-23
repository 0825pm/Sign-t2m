"""
extract_133d.py — 523D npy → 133D positions-only

523D에서 position 관련 columns만 추출:
  [0:4]     root_motion (rot_vel + linear_vel + root_y)
  [4:43]    body_ric (13 joints × 3, pelvis-relative)
  [163:208] lhand_ric (15 joints × 3, wrist-relative)
  [343:388] rhand_ric (15 joints × 3, wrist-relative)
  = 133D

Usage:
    python extract_133d.py                    # all datasets, all splits
    python extract_133d.py --dataset how2sign # how2sign only
"""

import os, argparse
import numpy as np
import torch
from tqdm import tqdm

BASE = '/home/user/Projects/research/SOKE/data'
FEAT_DIM = 133

# 523D → 133D index slices
IDX_SLICES = [
    slice(0, 4),      # root_motion
    slice(4, 43),     # body_ric
    slice(163, 208),  # lhand_ric
    slice(343, 388),  # rhand_ric
]

def extract_indices(data_523):
    """[T, 523] → [T, 133]"""
    return np.concatenate([data_523[:, s] for s in IDX_SLICES], axis=-1)


def process_dir(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith('.npy')]
    skip, done = 0, 0
    for f in tqdm(files, desc=os.path.basename(src_dir)):
        dst = os.path.join(dst_dir, f)
        if os.path.exists(dst):
            skip += 1
            continue
        data = np.load(os.path.join(src_dir, f))
        assert data.shape[1] == 523, f"{f}: shape={data.shape}"
        np.save(dst, extract_indices(data).astype(np.float32))
        done += 1
    print(f"  {done} saved, {skip} skipped")


def compute_stats(npy_dir, stat_dir):
    files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    if not files:
        return
    print(f"  Stats from {npy_dir} ({len(files)} files)")
    total, running_sum = 0, np.zeros(FEAT_DIM, dtype=np.float64)
    for f in tqdm(files, desc='mean', leave=False):
        d = np.load(os.path.join(npy_dir, f))
        running_sum += d.sum(axis=0).astype(np.float64)
        total += d.shape[0]
    mean = (running_sum / total).astype(np.float32)

    running_sq = np.zeros(FEAT_DIM, dtype=np.float64)
    for f in tqdm(files, desc='std', leave=False):
        d = np.load(os.path.join(npy_dir, f))
        running_sq += ((d - mean) ** 2).sum(axis=0).astype(np.float64)
    std = np.sqrt(running_sq / total).astype(np.float32)
    std = np.clip(std, 1e-6, None)

    os.makedirs(stat_dir, exist_ok=True)
    torch.save(torch.from_numpy(mean), os.path.join(stat_dir, 'mean_133.pt'))
    torch.save(torch.from_numpy(std), os.path.join(stat_dir, 'std_133.pt'))
    print(f"  ✅ {stat_dir}/mean_133.pt, std_133.pt ({total} frames)")

    # Summary
    parts = [('root_motion', 0, 4), ('body_ric', 4, 43),
             ('lhand_ric', 43, 88), ('rhand_ric', 88, 133)]
    for name, s, e in parts:
        m, st = mean[s:e], std[s:e]
        print(f"    {name:<12s} [{s:3d}:{e:3d}] mean=[{m.min():+.4f},{m.max():+.4f}] std=[{st.min():.4f},{st.max():.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=['how2sign', 'csl', 'phoenix', 'all'])
    args = parser.parse_args()

    datasets = ['how2sign', 'csl', 'phoenix'] if args.dataset == 'all' else [args.dataset]

    dirs_map = {
        'how2sign': {
            'src': f'{BASE}/How2Sign_523d', 'dst': f'{BASE}/How2Sign_133d',
            'splits': [('train/poses', 'train/poses'), ('val/poses', 'val/poses'), ('test/poses', 'test/poses')],
            'train_dir': 'train/poses',
        },
        'csl': {
            'src': f'{BASE}/CSL-Daily_523d', 'dst': f'{BASE}/CSL-Daily_133d',
            'splits': [('poses', 'poses')],
            'train_dir': 'poses',
        },
        'phoenix': {
            'src': f'{BASE}/Phoenix_523d', 'dst': f'{BASE}/Phoenix_133d',
            'splits': [('train', 'train'), ('val', 'val'), ('test', 'test')],
            'train_dir': 'train',
        },
    }

    for ds in datasets:
        info = dirs_map[ds]
        print(f"\n{'='*50}")
        print(f"  {ds}: 523D → 133D")
        print(f"{'='*50}")
        for src_sub, dst_sub in info['splits']:
            src = os.path.join(info['src'], src_sub)
            dst = os.path.join(info['dst'], dst_sub)
            if os.path.exists(src):
                process_dir(src, dst)

        train_dir = os.path.join(info['dst'], info['train_dir'])
        if os.path.exists(train_dir):
            compute_stats(train_dir, info['dst'])

    print("\n✅ Done!")


if __name__ == '__main__':
    main()
