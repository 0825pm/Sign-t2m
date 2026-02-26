"""
Diagnose body_loss explosion: inspect normalized pos107 data per dim.

Loads samples through the actual data pipeline, computes:
1. Per-dim stats (mean, std, min, max, |max|) of normalized data
2. Outlier counts (|z| > 5, |z| > 10, |z| > 50)
3. Per-dataset breakdown
4. Raw vs normalized comparison

Usage:
    python diagnose_body.py
"""
import numpy as np
import torch
import os
import glob

# === Same indexing as load_data.py ===
POS120_IDX = list(range(0, 30)) + list(range(90, 135)) + list(range(225, 270))
_ALIVE_BODY_LOCAL = [7, 10, 11, 13, 16] + list(range(18, 30))
ALIVE_POS_IDX = [POS120_IDX[i] for i in _ALIVE_BODY_LOCAL + list(range(30, 120))]

datasets = {
    'How2Sign': '/home/user/Projects/research/SOKE/data/data360/How2Sign',
    'CSL-Daily': '/home/user/Projects/research/SOKE/data/data360/CSL-Daily',
    'Phoenix_2014T': '/home/user/Projects/research/SOKE/data/data360/Phoenix_2014T',
}

print(f"ALIVE_POS_IDX ({len(ALIVE_POS_IDX)} dims):")
print(f"  body (17): {ALIVE_POS_IDX[:17]}")
print(f"  lhand (45): {ALIVE_POS_IDX[17:62]}")
print(f"  rhand (45): {ALIVE_POS_IDX[62:]}")
print()

for ds_name, base in datasets.items():
    print(f"\n{'='*60}")
    print(f"  {ds_name}")
    print(f"{'='*60}")

    # Load stats
    mean_path = f'{base}/mean_pos107.pt'
    std_path = f'{base}/std_pos107.pt'
    if not os.path.exists(mean_path):
        print(f"  [SKIP] stats not found: {mean_path}")
        continue
    mean = torch.load(mean_path).numpy()
    std = torch.load(std_path).numpy()

    print(f"\n--- Stats file inspection ---")
    print(f"  mean shape: {mean.shape}, std shape: {std.shape}")
    print(f"  Body dims [0:17]:")
    print(f"    mean: {mean[:17]}")
    print(f"    std:  {std[:17]}")
    print(f"    min std: {std[:17].min():.6f}, max std: {std[:17].max():.4f}")
    print(f"  Hand dims [17:107]:")
    print(f"    mean range: {mean[17:].min():.4f} ~ {mean[17:].max():.4f}")
    print(f"    std range:  {std[17:].min():.6f} ~ {std[17:].max():.4f}")

    # Load some actual data files
    files = sorted(glob.glob(f'{base}/train/*.npy'))[:200]  # sample 200
    if not files:
        print(f"  [SKIP] no npy files in {base}/train/")
        continue

    all_raw = []
    all_norm = []
    for f in files:
        data = np.load(f)
        if data.shape[1] < 270:
            print(f"  WARNING: {f} has shape {data.shape}, expected >= 270 cols")
            continue
        pos107 = data[:, ALIVE_POS_IDX].astype(np.float32)
        norm = (pos107 - mean) / np.maximum(std, 0.01)
        all_raw.append(pos107)
        all_norm.append(norm)

    all_raw = np.concatenate(all_raw, axis=0)
    all_norm = np.concatenate(all_norm, axis=0)
    print(f"\n--- Loaded {len(files)} files, {all_raw.shape[0]} frames ---")

    # Per-dim stats of NORMALIZED data
    print(f"\n--- Normalized data per-dim stats ---")
    print(f"{'dim':>4} {'part':>6} {'mean':>8} {'std':>8} {'min':>10} {'max':>10} {'|max|':>10} {'|>5|':>8} {'|>10|':>8} {'|>50|':>8}")
    print("-" * 95)

    for d in range(107):
        col = all_norm[:, d]
        part = "body" if d < 17 else ("lhand" if d < 62 else "rhand")
        n_gt5 = (np.abs(col) > 5).sum()
        n_gt10 = (np.abs(col) > 10).sum()
        n_gt50 = (np.abs(col) > 50).sum()
        absmax = np.abs(col).max()

        # Only print body dims (all 17) and summary for hands
        if d < 17 or n_gt50 > 0:
            print(f"{d:4d} {part:>6} {col.mean():8.3f} {col.std():8.3f} {col.min():10.3f} {col.max():10.3f} {absmax:10.3f} {n_gt5:8d} {n_gt10:8d} {n_gt50:8d}")

    # Summary
    body_norm = all_norm[:, :17]
    hand_norm = all_norm[:, 17:]
    print(f"\n--- Summary ---")
    print(f"  Body (17 dims): mean={body_norm.mean():.4f}, std={body_norm.std():.4f}, "
          f"|max|={np.abs(body_norm).max():.2f}, "
          f"|>5|={( np.abs(body_norm) > 5).sum()}, "
          f"|>10|={(np.abs(body_norm) > 10).sum()}, "
          f"|>50|={(np.abs(body_norm) > 50).sum()}")
    print(f"  Hand (90 dims): mean={hand_norm.mean():.4f}, std={hand_norm.std():.4f}, "
          f"|max|={np.abs(hand_norm).max():.2f}, "
          f"|>5|={(np.abs(hand_norm) > 5).sum()}, "
          f"|>10|={(np.abs(hand_norm) > 10).sum()}, "
          f"|>50|={(np.abs(hand_norm) > 50).sum()}")

    # Check raw data range
    print(f"\n--- Raw data range ---")
    print(f"  Body: {all_raw[:,:17].min():.4f} ~ {all_raw[:,:17].max():.4f}")
    print(f"  Hand: {all_raw[:,17:].min():.4f} ~ {all_raw[:,17:].max():.4f}")

    # Check if any 360D file has wrong shape
    print(f"\n--- Shape check (first 5 files) ---")
    for f in files[:5]:
        data = np.load(f)
        print(f"  {os.path.basename(f)}: shape={data.shape}")

print("\n\nDone.")
