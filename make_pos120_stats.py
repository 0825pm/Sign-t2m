"""
Compute pos120 mean/std from 360D npy data.

pos120 = body_pos(30) + lhand_pos(45) + rhand_pos(45) = 120 dims
Extracted from 360D via POS120_IDX.

std is floored at 0.01 for numerical stability (dead body dims).

Usage:
    python make_pos120_stats.py
"""
import torch
import numpy as np
import glob

# pos120 indices in 360D space
POS120_IDX = list(range(0, 30)) + list(range(90, 135)) + list(range(225, 270))

datasets = {
    'How2Sign': '/home/user/Projects/research/SOKE/data/data360/How2Sign',
    'CSL-Daily': '/home/user/Projects/research/SOKE/data/data360/CSL-Daily',
    'Phoenix_2014T': '/home/user/Projects/research/SOKE/data/data360/Phoenix_2014T',
}

for name, base in datasets.items():
    print(f'\n=== {name} ===')
    all_120 = []
    for split in ['train']:
        files = glob.glob(f'{base}/{split}/*.npy')
        print(f'  {split}: {len(files)} files')
        for f in files:
            data = np.load(f)
            if data.shape[1] >= 270:
                all_120.append(data[:, POS120_IDX])
            else:
                all_120.append(data[:, :120])

    all_120 = np.concatenate(all_120, axis=0)
    print(f'  Total frames: {all_120.shape[0]}, dims: {all_120.shape[1]}')

    mean = all_120.mean(axis=0)
    std = all_120.std(axis=0)

    print(f'  std range (raw): {std.min():.6f} ~ {std.max():.4f}')
    near_zero = (std < 0.01).sum()
    if near_zero > 0:
        print(f'  {near_zero} dims with std < 0.01 -> clamping to 0.01')
        print(f'  near-zero dim indices: {np.where(std < 0.01)[0].tolist()}')
    std = np.maximum(std, 0.01)
    print(f'  std range (clamped): {std.min():.6f} ~ {std.max():.4f}')

    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()

    torch.save(mean_t, f'{base}/mean_pos120.pt')
    torch.save(std_t, f'{base}/std_pos120.pt')
    print(f'  Saved: {base}/mean_pos120.pt, std_pos120.pt')
