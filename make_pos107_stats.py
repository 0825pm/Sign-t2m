"""
Compute clean position stats: remove dead (near-constant) body dims.

pos120 → pos107: 13 dead body dims removed (pelvis, spine, etc.)
Alive body dims (in pos120 space): 7, 10, 11, 13, 16, 18-29
Hand dims (45+45) all alive.

Usage:
    python make_pos107_stats.py
"""
import torch
import numpy as np
import glob
import os

# Original pos120 indices in 360D space
POS120_IDX = list(range(0, 30)) + list(range(90, 135)) + list(range(225, 270))

# Alive body dims (in pos120 local space, i.e., within the 30 body dims)
# std >= 0.01 from How2Sign train set analysis
ALIVE_BODY_LOCAL = [7, 10, 11, 13, 16] + list(range(18, 30))  # 17 dims
ALIVE_HAND_LOCAL = list(range(30, 120))                         # 90 dims
ALIVE_LOCAL = ALIVE_BODY_LOCAL + ALIVE_HAND_LOCAL               # 107 dims

# Convert to 360D space indices
ALIVE_POS_IDX = [POS120_IDX[i] for i in ALIVE_LOCAL]

print(f"pos120 → pos107: {len(POS120_IDX)} → {len(ALIVE_POS_IDX)} dims")
print(f"  body: 30 → {len(ALIVE_BODY_LOCAL)}")
print(f"  hand: 90 → 90 (unchanged)")
print(f"  ALIVE_POS_IDX = {ALIVE_POS_IDX}")

datasets = {
    'How2Sign': '/home/user/Projects/research/SOKE/data/data360/How2Sign',
    'CSL-Daily': '/home/user/Projects/research/SOKE/data/data360/CSL-Daily',
    'Phoenix_2014T': '/home/user/Projects/research/SOKE/data/data360/Phoenix_2014T',
}

for name, base in datasets.items():
    print(f'\n=== {name} ===')
    all_frames = []
    for split in ['train']:
        files = glob.glob(f'{base}/{split}/*.npy')
        print(f'  {split}: {len(files)} files')
        for f in files:
            data = np.load(f)
            pos107 = data[:, ALIVE_POS_IDX]
            all_frames.append(pos107)

    all_frames = np.concatenate(all_frames, axis=0)
    print(f'  Total frames: {all_frames.shape[0]}, dims: {all_frames.shape[1]}')

    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)

    print(f'  std range (raw): {std.min():.6f} ~ {std.max():.4f}')
    near_zero = (std < 0.01).sum()
    if near_zero > 0:
        print(f'  WARNING: {near_zero} dims with std < 0.01 → clamping to 0.01')
        print(f'  near-zero dim indices: {np.where(std < 0.01)[0].tolist()}')
    std = np.maximum(std, 0.01)
    print(f'  std range (clamped): {std.min():.6f} ~ {std.max():.4f}')

    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()

    torch.save(mean_t, f'{base}/mean_pos107.pt')
    torch.save(std_t, f'{base}/std_pos107.pt')
    print(f'  Saved: {base}/mean_pos107.pt, std_pos107.pt')
