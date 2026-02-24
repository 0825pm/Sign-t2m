"""
compute_stats_133d.py — SOKE pkl에서 직접 133D mean/std 계산

Usage:
    cd ~/Projects/research/sign-t2m
    python compute_stats_133d.py                    # How2Sign + CSL + Phoenix
    python compute_stats_133d.py --dataset how2sign # How2Sign만
"""

import os, sys, argparse
import numpy as np
import torch
from tqdm import tqdm

torch.backends.cudnn.enabled = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.signlang.dataset_sign import SignMotionDataset

BASE = '/home/user/Projects/research/SOKE/data'

DATASETS = {
    'how2sign': dict(
        data_root=f'{BASE}/How2Sign',
        dataset_name='how2sign',
        split='train',
        out_dir=f'{BASE}/How2Sign',
    ),
    'csl': dict(
        data_root=f'{BASE}/How2Sign',   # dummy (not used)
        csl_root=f'{BASE}/CSL-Daily',
        dataset_name='csl',
        split='train',
        out_dir=f'{BASE}/CSL-Daily',
    ),
    'phoenix': dict(
        data_root=f'{BASE}/How2Sign',   # dummy
        phoenix_root=f'{BASE}/Phoenix_2014T',
        dataset_name='phoenix',
        split='train',
        out_dir=f'{BASE}/Phoenix_2014T',
    ),
}


def compute_stats(dataset_name, cfg):
    print(f"\n[{dataset_name}] 133D mean/std 계산 중...")

    # dummy mean/std (정규화 전 로딩용)
    dummy_mean = torch.zeros(133)
    dummy_std  = torch.ones(133)

    ds = SignMotionDataset(
        data_root=cfg['data_root'],
        csl_root=cfg.get('csl_root'),
        phoenix_root=cfg.get('phoenix_root'),
        split=cfg['split'],
        mean=dummy_mean,
        std=dummy_std,
        nfeats=133,
        dataset_name=cfg['dataset_name'],
        max_motion_length=10000,   # clip 없이 전체 로드
        min_motion_length=1,
    )
    print(f"  Loaded {len(ds)} samples")

    # running stats (두 패스)
    total_frames = 0
    running_sum  = np.zeros(133, dtype=np.float64)

    for i in tqdm(range(len(ds)), desc='Pass 1/2 (mean)'):
        item = ds[i]
        # denormalize back (dummy norm이라 그냥 그대로)
        motion = item['motion'].numpy()   # [T, 133] — dummy std=1, mean=0이라 raw값
        running_sum += motion.sum(axis=0).astype(np.float64)
        total_frames += motion.shape[0]

    mean = (running_sum / total_frames).astype(np.float32)

    running_sq = np.zeros(133, dtype=np.float64)
    for i in tqdm(range(len(ds)), desc='Pass 2/2 (std)'):
        item = ds[i]
        motion = item['motion'].numpy()
        running_sq += ((motion - mean) ** 2).sum(axis=0).astype(np.float64)

    std = np.sqrt(running_sq / total_frames).astype(np.float32)
    std = np.clip(std, 1e-6, None)

    out = cfg['out_dir']
    os.makedirs(out, exist_ok=True)
    torch.save(torch.from_numpy(mean), os.path.join(out, 'mean_133.pt'))
    torch.save(torch.from_numpy(std),  os.path.join(out, 'std_133.pt'))
    print(f"  ✅ Saved → {out}/mean_133.pt, std_133.pt  ({total_frames} frames)")

    # 파트별 통계
    parts = [('body+upper', 0, 30), ('lhand', 30, 75), ('rhand', 75, 120),
             ('jaw', 120, 123), ('expr', 123, 133)]
    for name, s, e in parts:
        print(f"    {name:<12s} [{s:3d}:{e:3d}] "
              f"mean=[{mean[s:e].min():+.4f},{mean[s:e].max():+.4f}] "
              f"std=[{std[s:e].min():.4f},{std[s:e].max():.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all',
                        choices=['how2sign', 'csl', 'phoenix', 'all'])
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    for ds in targets:
        compute_stats(ds, DATASETS[ds])

    print("\n✅ 완료! 이제 학습 시작 가능")


if __name__ == '__main__':
    main()