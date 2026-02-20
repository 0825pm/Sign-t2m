"""
전체 데이터셋 시퀀스 길이 분포 + threshold 비교
How2Sign / CSL-Daily / Phoenix × train / val / test

Usage:
    python check_lengths_all.py
"""
import os, gzip, pickle, math, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H2S_ROOT     = "/home/user/Projects/research/SOKE/data/How2Sign"
CSL_ROOT     = "/home/user/Projects/research/SOKE/data/CSL-Daily"
PHOENIX_ROOT = "/home/user/Projects/research/SOKE/data/Phoenix_2014T"

BAD_IDS = {
    '0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front',
    '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front',
    '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front',
    '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front',
    '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front',
    'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front',
    'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front',
    'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front',
    'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front',
    '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front',
    'g3Cc_1-V31U_12-3-rgb_front',
}

def get_h2s_lengths(split):
    csv_path = os.path.join(H2S_ROOT, split, 're_aligned',
                            f'how2sign_realigned_{split}_preprocessed_fps.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(H2S_ROOT, split, 'preprocessed_fps.csv')
    if not os.path.exists(csv_path): return []
    csv = pd.read_csv(csv_path)
    csv['DURATION'] = csv['END_REALIGNED'] - csv['START_REALIGNED']
    csv = csv[csv['DURATION'] < 30].reset_index(drop=True)
    lengths = []
    for idx in tqdm(range(len(csv)), desc=f'  H2S-{split}', leave=False):
        name = csv.iloc[idx]['SENTENCE_NAME']
        fps = csv.iloc[idx]['fps']
        if name in BAD_IDS: continue
        pose_dir = os.path.join(H2S_ROOT, split, 'poses', name)
        if not os.path.exists(pose_dir):
            pose_dir = os.path.join(H2S_ROOT, 'poses', name)
        if not os.path.exists(pose_dir): continue
        n_raw = len([f for f in os.listdir(pose_dir) if f.endswith('.pkl')])
        n_final = int(24 * n_raw / fps) if fps > 24 else n_raw
        if n_final >= 1: lengths.append(n_final)
    return lengths

def get_csl_lengths(split):
    ann_path = os.path.join(CSL_ROOT, f'csl_clean.{split}')
    if not os.path.exists(ann_path): return []
    with gzip.open(ann_path, 'rb') as f: ann = pickle.load(f)
    lengths = []
    for item in tqdm(ann, desc=f'  CSL-{split}', leave=False):
        pose_dir = os.path.join(CSL_ROOT, 'poses', item['name'])
        if not os.path.exists(pose_dir): continue
        n = len([f for f in os.listdir(pose_dir) if f.endswith('.pkl')])
        if n >= 1: lengths.append(n)
    return lengths

def get_phoenix_lengths(split):
    if split == 'val':
        ann_path = os.path.join(PHOENIX_ROOT, 'phoenix14t.dev')
    else:
        ann_path = os.path.join(PHOENIX_ROOT, f'phoenix14t.{split}')
    if not os.path.exists(ann_path): return []
    with gzip.open(ann_path, 'rb') as f: ann = pickle.load(f)
    lengths = []
    for item in tqdm(ann, desc=f'  Phx-{split}', leave=False):
        pose_dir = os.path.join(PHOENIX_ROOT, item['name'])
        if not os.path.exists(pose_dir): continue
        n = len([f for f in os.listdir(pose_dir) if f.endswith('.pkl')])
        if n >= 1: lengths.append(n)
    return lengths


def main():
    datasets = ['How2Sign', 'CSL-Daily', 'Phoenix']
    splits = ['train', 'val', 'test']
    get_fns = {'How2Sign': get_h2s_lengths, 'CSL-Daily': get_csl_lengths, 'Phoenix': get_phoenix_lengths}

    # ── 1. 데이터 수집 ──
    data = {}
    for ds in datasets:
        data[ds] = {}
        print(f"\nCollecting {ds}...")
        for sp in splits:
            data[ds][sp] = np.array(get_fns[ds](sp))
            n = len(data[ds][sp])
            if n > 0:
                f = data[ds][sp]
                print(f"  {sp}: n={n}  [{f.min()}, {f.max()}]  mean={f.mean():.1f}  med={np.median(f):.0f}")

    # ── 2. Threshold 비교 ──
    configs = [
        (40, 300, "현재 (40-300)"),
        (30, 300, "min=30"),
        (20, 300, "min=20"),
        (40, 400, "max=400"),
        (30, 400, "30-400"),
        (20, 400, "20-400"),
        (20, 500, "20-500"),
    ]

    for ds in datasets:
        print(f"\n{'='*95}")
        print(f"  {ds} — Threshold 비교")
        print(f"{'='*95}")
        print(f"  {'Config':<18s}", end="")
        for sp in splits:
            print(f"  │ {sp:>5s} in-rng  <min  >max", end="")
        print()
        print(f"  {'-'*18}", end="")
        for _ in splits:
            print(f"--+{'-'*27}", end="")
        print()

        for mn, mx, label in configs:
            row = f"  {label:<18s}"
            for sp in splits:
                f = data[ds][sp]
                if len(f) == 0:
                    row += f"  │  N/A"; continue
                below = (f < mn).sum()
                above = (f > mx).sum()
                pct = ((f >= mn) & (f <= mx)).mean() * 100
                row += f"  │ {pct:>5.1f}% {below:>5d} {above:>5d}"
            print(row)

    # ── 3. 그래프 ──
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Sequence Length Distribution', fontsize=14, fontweight='bold')
    colors = {'How2Sign': 'steelblue', 'CSL-Daily': 'seagreen', 'Phoenix': 'coral'}

    for row, ds in enumerate(datasets):
        for col, sp in enumerate(splits):
            ax = axes[row][col]
            f = data[ds][sp]
            if len(f) == 0:
                ax.set_title(f'{ds}/{sp} (no data)'); ax.axis('off'); continue
            max_x = min(800, f.max() + 50)
            ax.hist(f, bins=60, range=(0, max_x), color=colors[ds],
                    edgecolor='black', linewidth=0.3, alpha=0.8)
            ax.axvline(x=40, color='red', linestyle='--', linewidth=1.5, label='40')
            ax.axvline(x=300, color='red', linestyle='--', linewidth=1.5, label='300')
            ax.axvline(x=400, color='orange', linestyle=':', linewidth=1.5, label='400')
            ax.axvline(x=np.median(f), color='black', linestyle='-', linewidth=1.5, alpha=0.7)
            in_cur = ((f >= 40) & (f <= 300)).mean() * 100
            in_new = ((f >= 20) & (f <= 400)).mean() * 100
            ax.set_title(f'{ds}/{sp} (n={len(f)})', fontsize=11)
            ax.text(0.97, 0.95,
                    f'40-300: {in_cur:.0f}%\n20-400: {in_new:.0f}%\nmed: {np.median(f):.0f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
            if col == 0: ax.set_ylabel('Count')
            if row == 2: ax.set_xlabel('Frames')

    plt.tight_layout()
    plt.savefig('seq_length_all.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: seq_length_all.png")


if __name__ == "__main__":
    main()