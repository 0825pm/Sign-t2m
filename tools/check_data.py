"""
데이터 전처리 종합 진단

체크 항목:
1. mean/std 자체 분석 (차원별 값, near-zero std)
2. 정규화 후 분포 (mean~0, std~1이 되는지)
3. 이상치/NaN/Inf 검출
4. 파트별(body/lhand/rhand) 분포 비교
5. 시간축 연속성 (프레임간 급변 체크)
6. 실제 샘플 로딩 실패율

Usage:
    python check_data.py
"""
import os, sys, pickle, math, re
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

DATA_ROOT = "/home/user/Projects/research/SOKE/data/How2Sign"
MEAN_PATH = "/home/user/Projects/research/SOKE/data/CSL-Daily/mean_120.pt"
STD_PATH  = "/home/user/Projects/research/SOKE/data/CSL-Daily/std_120.pt"
NFEATS = 120
MAX_SAMPLES = 500  # 빠른 진단용 (전체: -1)

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


def extract_frame_num(fn):
    m = re.search(r'_(\d+)_3D\.pkl$', fn)
    if m: return int(m.group(1))
    m = re.search(r'_?(\d+)\.pkl$', fn)
    if m: return int(m.group(1))
    nums = re.findall(r'\d+', fn)
    return int(nums[-1]) if nums else 0


def get_pose(d):
    SOKE = ['smplx_root_pose','smplx_body_pose','smplx_lhand_pose',
            'smplx_rhand_pose','smplx_jaw_pose','smplx_shape','smplx_expr']
    NEW  = ['global_orient','body_pose','left_hand_pose',
            'right_hand_pose','jaw_pose','betas','expression']
    vals = []
    if all(k in d for k in SOKE):
        for k in SOKE: vals.append(np.array(d[k]).flatten())
    else:
        for nk, sk in zip(NEW, SOKE):
            if nk in d: vals.append(np.array(d[nk]).flatten())
            elif sk in d: vals.append(np.array(d[sk]).flatten())
            else: return None
    return np.concatenate(vals) if vals else None


def main():
    # ── 1. mean/std 분석 ──
    print("=" * 70)
    print("1. MEAN / STD 파일 분석")
    print("=" * 70)

    mean = torch.load(MEAN_PATH, map_location='cpu').numpy()
    std = torch.load(STD_PATH, map_location='cpu').numpy()

    print(f"  mean shape: {mean.shape}")
    print(f"  std  shape: {std.shape}")

    # 120D만 사용
    mean = mean[:NFEATS]
    std = std[:NFEATS]

    parts = [("body[0:30]", 0, 30), ("lhand[30:75]", 30, 75), ("rhand[75:120]", 75, 120)]
    for name, s, e in parts:
        m, st = mean[s:e], std[s:e]
        print(f"\n  {name}:")
        print(f"    mean: min={m.min():.6f}  max={m.max():.6f}  avg={m.mean():.6f}")
        print(f"    std:  min={st.min():.6f}  max={st.max():.6f}  avg={st.mean():.6f}")
        near_zero = (st < 0.01).sum()
        very_large = (st > 5.0).sum()
        print(f"    near-zero std (<0.01): {near_zero} dims")
        print(f"    very large std (>5.0): {very_large} dims")
        if near_zero > 0:
            idxs = np.where(st < 0.01)[0]
            print(f"    ⚠️  near-zero std dims: {[s+i for i in idxs]}")
            print(f"       these dimensions will be amplified ~100x+ after normalization!")

    # ── 2. raw 데이터 로드 및 분석 ──
    print(f"\n{'='*70}")
    print("2. RAW DATA 로딩 + 정규화 후 분포 분석")
    print("=" * 70)

    csv_path = os.path.join(DATA_ROOT, 'train', 're_aligned',
                            'how2sign_realigned_train_preprocessed_fps.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATA_ROOT, 'train', 'preprocessed_fps.csv')
    csv = pd.read_csv(csv_path)
    csv['DURATION'] = csv['END_REALIGNED'] - csv['START_REALIGNED']
    csv = csv[csv['DURATION'] < 30].reset_index(drop=True)

    n = min(MAX_SAMPLES, len(csv)) if MAX_SAMPLES > 0 else len(csv)
    print(f"  Checking {n}/{len(csv)} samples...\n")

    all_raw_frames = []
    all_norm_frames = []
    all_velocities = []
    load_failures = 0
    zero_frame_clips = 0
    nan_clips = 0
    sample_lengths = []
    frame_jump_clips = 0  # 프레임간 급격한 변화

    for idx in tqdm(range(n), desc="Loading"):
        name = csv.iloc[idx]['SENTENCE_NAME']
        fps = csv.iloc[idx]['fps']
        if name in BAD_IDS:
            continue

        pose_dir = os.path.join(DATA_ROOT, 'train', 'poses', name)
        if not os.path.exists(pose_dir):
            load_failures += 1
            continue

        pkls = sorted(
            [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
            key=extract_frame_num
        )

        # FPS subsample (like dataset)
        if fps > 24:
            target = int(24 * len(pkls) / fps)
            ss = float(len(pkls)) / target
            pkls = [pkls[int(math.floor(i * ss))] for i in range(target)]

        if len(pkls) < 4:
            load_failures += 1
            continue

        clip = np.zeros([len(pkls), 179], dtype=np.float32)
        for fi, frame in enumerate(pkls):
            try:
                with open(os.path.join(pose_dir, frame), 'rb') as f:
                    d = pickle.load(f)
                p = get_pose(d)
                if p is not None and len(p) >= 179:
                    clip[fi] = p[:179]
            except:
                continue

        # 179 → 120
        raw_120 = clip[:, 36:156]
        sample_lengths.append(len(raw_120))

        # all-zero frame 체크
        zero_rows = (np.abs(raw_120).sum(axis=1) < 1e-8).sum()
        if zero_rows > 0:
            zero_frame_clips += 1

        # NaN/Inf 체크
        if np.any(np.isnan(raw_120)) or np.any(np.isinf(raw_120)):
            nan_clips += 1
            continue

        all_raw_frames.append(raw_120)

        # 정규화
        norm = (raw_120 - mean) / (std + 1e-10)
        all_norm_frames.append(norm)

        # velocity (프레임 간 차이)
        if len(norm) > 1:
            vel = np.diff(norm, axis=0)
            all_velocities.append(vel)
            max_jump = np.abs(vel).max()
            if max_jump > 10:
                frame_jump_clips += 1

    print(f"\n  Load failures: {load_failures}/{n}")
    print(f"  Zero-frame clips: {zero_frame_clips}")
    print(f"  NaN/Inf clips: {nan_clips}")
    print(f"  Frame-jump clips (max|vel|>10): {frame_jump_clips}")

    # ── 3. 정규화 후 분포 ──
    if len(all_norm_frames) == 0:
        print("ERROR: No data loaded!")
        return

    raw_all = np.concatenate(all_raw_frames, axis=0)
    norm_all = np.concatenate(all_norm_frames, axis=0)
    vel_all = np.concatenate(all_velocities, axis=0) if all_velocities else np.zeros((1, NFEATS))

    print(f"\n{'='*70}")
    print("3. 정규화 후 분포 (이상적: mean~0, std~1)")
    print("=" * 70)
    print(f"  Total frames analyzed: {len(norm_all)}")

    for name, s, e in parts:
        raw_part = raw_all[:, s:e]
        norm_part = norm_all[:, s:e]
        vel_part = vel_all[:, s:e]

        print(f"\n  {name}:")
        print(f"    [RAW]  mean={raw_part.mean():.4f}  std={raw_part.std():.4f}  "
              f"range=[{raw_part.min():.3f}, {raw_part.max():.3f}]")
        print(f"    [NORM] mean={norm_part.mean():.4f}  std={norm_part.std():.4f}  "
              f"range=[{norm_part.min():.3f}, {norm_part.max():.3f}]")
        print(f"    [VEL]  mean={vel_part.mean():.4f}  std={vel_part.std():.4f}  "
              f"max|vel|={np.abs(vel_part).max():.3f}")

        # 차원별 분석
        dim_means = norm_part.mean(axis=0)
        dim_stds = norm_part.std(axis=0)
        bad_mean = (np.abs(dim_means) > 1.0).sum()
        bad_std_low = (dim_stds < 0.1).sum()
        bad_std_high = (dim_stds > 5.0).sum()

        if bad_mean > 0:
            print(f"    ⚠️  {bad_mean} dims with |norm_mean| > 1.0: {np.where(np.abs(dim_means) > 1.0)[0] + s}")
        if bad_std_low > 0:
            print(f"    ⚠️  {bad_std_low} dims with norm_std < 0.1: {np.where(dim_stds < 0.1)[0] + s}")
        if bad_std_high > 0:
            print(f"    ⚠️  {bad_std_high} dims with norm_std > 5.0: {np.where(dim_stds > 5.0)[0] + s}")

        # 이상치 비율
        outlier_3 = (np.abs(norm_part) > 3).mean() * 100
        outlier_5 = (np.abs(norm_part) > 5).mean() * 100
        outlier_10 = (np.abs(norm_part) > 10).mean() * 100
        print(f"    Outliers: >3σ={outlier_3:.2f}%  >5σ={outlier_5:.3f}%  >10σ={outlier_10:.4f}%")

    # ── 4. 시퀀스 길이 분포 ──
    print(f"\n{'='*70}")
    print("4. 시퀀스 길이 분포")
    print("=" * 70)
    lens = np.array(sample_lengths)
    print(f"  count: {len(lens)}")
    print(f"  min={lens.min()}  max={lens.max()}  mean={lens.mean():.1f}  median={np.median(lens):.0f}")
    print(f"  <40 frames: {(lens < 40).sum()}")
    print(f"  >300 frames: {(lens > 300).sum()}")

    # ── 5. 차원별 상세 (가장 문제될 수 있는 차원들) ──
    print(f"\n{'='*70}")
    print("5. 차원별 상세 (정규화 후 이상 차원)")
    print("=" * 70)

    dim_means = norm_all.mean(axis=0)
    dim_stds = norm_all.std(axis=0)

    # 가장 편향된 10개 차원
    bias_order = np.argsort(np.abs(dim_means))[::-1]
    print("\n  Top 10 biased dims (|norm_mean| highest):")
    for i in bias_order[:10]:
        part = "body" if i < 30 else ("lhand" if i < 75 else "rhand")
        print(f"    dim {i:3d} ({part:5s}): norm_mean={dim_means[i]:+.4f}  norm_std={dim_stds[i]:.4f}  "
              f"raw_mean={mean[i]:.4f}  raw_std={std[i]:.6f}")

    # std가 가장 낮은 10개 차원
    std_order = np.argsort(dim_stds)
    print("\n  Top 10 lowest-variance dims (norm_std lowest):")
    for i in std_order[:10]:
        part = "body" if i < 30 else ("lhand" if i < 75 else "rhand")
        print(f"    dim {i:3d} ({part:5s}): norm_std={dim_stds[i]:.4f}  raw_std={std[i]:.6f}  "
              f"norm_mean={dim_means[i]:+.4f}")

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
