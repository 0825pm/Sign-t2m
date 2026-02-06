#!/usr/bin/env python
"""
6D Rotation 데이터 전처리 스크립트

기존 pkl (179-dim) → 새 npy (240-dim 6D rotation) 변환
+ mean_240.pt, std_240.pt 생성

파이프라인:
    1. pkl 로드 (179-dim)
    2. 슬라이싱 179 → 120 (axis-angle)
    3. 6D 변환 120 → 240
    4. 새 디렉토리에 .npy 저장
    5. 전체 데이터에서 mean_240, std_240 계산

사용법:
    python preprocess_6d.py --src_root /path/to/SOKE/data --dst_root /path/to/SOKE/data6d
    
    # 테스트 (일부만):
    python preprocess_6d.py --src_root /path/to/SOKE/data --dst_root /path/to/SOKE/data6d --max_samples 100
"""

import os
import sys
import re
import argparse
import pickle
import gzip
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# ============================================================
# SMPLX Keys
# ============================================================
SOKE_KEYS = [
    'smplx_root_pose',    # 3
    'smplx_body_pose',    # 63
    'smplx_lhand_pose',   # 45
    'smplx_rhand_pose',   # 45
    'smplx_jaw_pose',     # 3
    'smplx_shape',        # 10
    'smplx_expr'          # 10
]

NEW_KEYS = [
    ('global_orient', 'smplx_root_pose'),
    ('body_pose', 'smplx_body_pose'),
    ('left_hand_pose', 'smplx_lhand_pose'),
    ('right_hand_pose', 'smplx_rhand_pose'),
    ('jaw_pose', 'smplx_jaw_pose'),
    ('betas', 'smplx_shape'),
    ('expression', 'smplx_expr')
]


# ============================================================
# 6D Rotation 변환 함수 (Vectorized)
# ============================================================

def axis_angle_to_6d(axis_angle):
    """
    Axis-angle [T, J, 3] → 6D rotation [T, J, 6]
    Vectorized implementation
    """
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)  # [T, J, 1]
    
    safe_angle = np.where(angle < 1e-8, 1.0, angle)
    axis = axis_angle / safe_angle  # [T, J, 3]
    
    c = np.cos(angle)        # [T, J, 1]
    s = np.sin(angle)        # [T, J, 1]
    omc = 1 - c              # [T, J, 1]
    
    kx = axis[..., 0:1]
    ky = axis[..., 1:2]
    kz = axis[..., 2:3]
    
    r00 = c + kx * kx * omc
    r10 = ky * kx * omc + kz * s
    r20 = kz * kx * omc - ky * s
    r01 = kx * ky * omc - kz * s
    r11 = c + ky * ky * omc
    r21 = kz * ky * omc + kx * s
    
    rot_6d = np.concatenate([r00, r10, r20, r01, r11, r21], axis=-1)
    
    small_angle_mask = (angle < 1e-8).squeeze(-1)
    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    rot_6d[small_angle_mask] = identity_6d
    
    return rot_6d.astype(np.float32)


def convert_179_to_240(motion_179):
    """
    179-dim → 240-dim 6D rotation
    
    Steps:
        1. 179 → 120 (슬라이싱)
        2. 120 → 240 (6D 변환)
    """
    # 1. 슬라이싱: 179 → 120
    # [36:156] = upper_body(30) + lhand(45) + rhand(45)
    motion_120 = motion_179[:, 36:156]  # [T, 120]
    
    # 2. 6D 변환: 120 → 240
    T = motion_120.shape[0]
    motion_joints = motion_120.reshape(T, 40, 3)  # [T, 40, 3]
    motion_6d = axis_angle_to_6d(motion_joints)    # [T, 40, 6]
    
    return motion_6d.reshape(T, 240)  # [T, 240]


# ============================================================
# pkl 로딩 함수
# ============================================================

def extract_frame_num(filename):
    """파일명에서 프레임 번호 추출"""
    match = re.search(r'_(\d+)_3D\.pkl$', filename)
    if match:
        return int(match.group(1))
    
    match = re.search(r'_?(\d+)\.pkl$', filename)
    if match:
        return int(match.group(1))
    
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])
    
    return 0


def get_pose_from_pkl(poses_dict):
    """pkl에서 179-dim pose 추출"""
    pose_values = []
    
    # SOKE keys 먼저 시도
    if all(key in poses_dict for key in SOKE_KEYS):
        for key in SOKE_KEYS:
            val = np.array(poses_dict[key]).flatten()
            pose_values.append(val)
    else:
        # New keys 시도
        for new_key, soke_key in NEW_KEYS:
            if new_key in poses_dict:
                val = np.array(poses_dict[new_key]).flatten()
                pose_values.append(val)
            elif soke_key in poses_dict:
                val = np.array(poses_dict[soke_key]).flatten()
                pose_values.append(val)
            else:
                return None
    
    if pose_values:
        return np.concatenate(pose_values)
    return None


def load_sequence_from_pkls(pose_dir):
    """
    폴더 내 모든 pkl 파일 → [T, 179] numpy array
    """
    if not os.path.exists(pose_dir):
        return None
    
    pkl_files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    
    if len(pkl_files) < 4:
        return None
    
    motion_179 = np.zeros([len(pkl_files), 179], dtype=np.float32)
    
    for frame_id, frame in enumerate(pkl_files):
        frame_path = os.path.join(pose_dir, frame)
        try:
            with open(frame_path, 'rb') as f:
                poses_dict = pickle.load(f)
            pose = get_pose_from_pkl(poses_dict)
            if pose is not None and len(pose) >= 179:
                motion_179[frame_id] = pose[:179]
            elif pose is not None:
                motion_179[frame_id, :len(pose)] = pose
        except:
            continue
    
    return motion_179


# ============================================================
# 데이터셋별 처리
# ============================================================

def process_how2sign(src_root, dst_root, split, max_samples=None):
    """How2Sign 처리"""
    src_dir = os.path.join(src_root, 'How2Sign', split, 'poses')
    dst_dir = os.path.join(dst_root, 'How2Sign', split)
    os.makedirs(dst_dir, exist_ok=True)
    
    if not os.path.exists(src_dir):
        print(f"  [How2Sign/{split}] Source not found: {src_dir}")
        return []
    
    video_names = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    if max_samples:
        video_names = video_names[:max_samples]
    
    all_motions = []
    saved_count = 0
    
    for name in tqdm(video_names, desc=f'How2Sign/{split}', leave=False):
        pose_dir = os.path.join(src_dir, name)
        motion_179 = load_sequence_from_pkls(pose_dir)
        
        if motion_179 is None:
            continue
        
        # 변환: 179 → 240
        motion_240 = convert_179_to_240(motion_179)
        
        # 유효성 검사
        if np.isnan(motion_240).any() or np.isinf(motion_240).any():
            continue
        
        # 저장
        save_path = os.path.join(dst_dir, f'{name}.npy')
        np.save(save_path, motion_240)
        
        all_motions.append(motion_240)
        saved_count += 1
    
    print(f"  [How2Sign/{split}] Saved {saved_count} sequences")
    return all_motions


def process_csl(src_root, dst_root, split, max_samples=None):
    """CSL-Daily 처리"""
    src_pose_dir = os.path.join(src_root, 'CSL-Daily', 'poses')
    dst_dir = os.path.join(dst_root, 'CSL-Daily', split)
    os.makedirs(dst_dir, exist_ok=True)
    
    # Annotation 로드
    ann_path = os.path.join(src_root, 'CSL-Daily', f'csl_clean.{split}')
    if not os.path.exists(ann_path):
        print(f"  [CSL-Daily/{split}] Annotation not found: {ann_path}")
        return []
    
    try:
        with gzip.open(ann_path, 'rb') as f:
            annotations = pickle.load(f)
    except:
        print(f"  [CSL-Daily/{split}] Failed to load annotation")
        return []
    
    if max_samples:
        annotations = annotations[:max_samples]
    
    all_motions = []
    saved_count = 0
    
    for ann in tqdm(annotations, desc=f'CSL-Daily/{split}', leave=False):
        name = ann['name']
        pose_dir = os.path.join(src_pose_dir, name)
        motion_179 = load_sequence_from_pkls(pose_dir)
        
        if motion_179 is None:
            continue
        
        # 변환: 179 → 240
        motion_240 = convert_179_to_240(motion_179)
        
        # 유효성 검사
        if np.isnan(motion_240).any() or np.isinf(motion_240).any():
            continue
        
        # 저장
        save_path = os.path.join(dst_dir, f'{name}.npy')
        np.save(save_path, motion_240)
        
        all_motions.append(motion_240)
        saved_count += 1
    
    print(f"  [CSL-Daily/{split}] Saved {saved_count} sequences")
    return all_motions


def process_phoenix(src_root, dst_root, split, max_samples=None):
    """Phoenix-2014T 처리"""
    src_pose_dir = os.path.join(src_root, 'Phoenix_2014T')
    dst_dir = os.path.join(dst_root, 'Phoenix_2014T', split)
    os.makedirs(dst_dir, exist_ok=True)
    
    # Annotation 로드
    ann_name = 'phoenix14t.dev' if split == 'val' else f'phoenix14t.{split}'
    ann_path = os.path.join(src_root, 'Phoenix_2014T', ann_name)
    
    if not os.path.exists(ann_path):
        print(f"  [Phoenix/{split}] Annotation not found: {ann_path}")
        return []
    
    try:
        with gzip.open(ann_path, 'rb') as f:
            annotations = pickle.load(f)
    except:
        print(f"  [Phoenix/{split}] Failed to load annotation")
        return []
    
    if max_samples:
        annotations = annotations[:max_samples]
    
    all_motions = []
    saved_count = 0
    
    for ann in tqdm(annotations, desc=f'Phoenix/{split}', leave=False):
        name = ann['name']
        pose_dir = os.path.join(src_pose_dir, name)
        motion_179 = load_sequence_from_pkls(pose_dir)
        
        if motion_179 is None:
            continue
        
        # 변환: 179 → 240
        motion_240 = convert_179_to_240(motion_179)
        
        # 유효성 검사
        if np.isnan(motion_240).any() or np.isinf(motion_240).any():
            continue
        
        # name에서 split prefix 제거 (train/xxx → xxx)
        # Phoenix name이 'train/video_name' 형태일 수 있음
        save_name = name.split('/')[-1] if '/' in name else name
        
        # 저장
        save_path = os.path.join(dst_dir, f'{save_name}.npy')
        np.save(save_path, motion_240)
        
        all_motions.append(motion_240)
        saved_count += 1
    
    print(f"  [Phoenix/{split}] Saved {saved_count} sequences")
    return all_motions


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Preprocess 6D rotation data')
    parser.add_argument('--src_root', type=str, required=True,
                        help='Source data root (e.g., /path/to/SOKE/data)')
    parser.add_argument('--dst_root', type=str, required=True,
                        help='Destination data root (e.g., /path/to/SOKE/data6d)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per split (for testing)')
    parser.add_argument('--datasets', type=str, default='how2sign,csl,phoenix',
                        help='Datasets to process (comma-separated)')
    parser.add_argument('--splits', type=str, default='train,val,test',
                        help='Splits to process (comma-separated)')
    args = parser.parse_args()
    
    datasets = [d.strip().lower() for d in args.datasets.split(',')]
    splits = [s.strip() for s in args.splits.split(',')]
    
    print("=" * 60)
    print("6D Rotation Data Preprocessing")
    print("=" * 60)
    print(f"Source: {args.src_root}")
    print(f"Destination: {args.dst_root}")
    print(f"Datasets: {datasets}")
    print(f"Splits: {splits}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("=" * 60)
    
    os.makedirs(args.dst_root, exist_ok=True)
    
    all_motions = []
    
    # Process each dataset and split
    for split in splits:
        print(f"\n[Processing {split}]")
        
        if 'how2sign' in datasets:
            motions = process_how2sign(args.src_root, args.dst_root, split, args.max_samples)
            all_motions.extend(motions)
        
        if 'csl' in datasets:
            motions = process_csl(args.src_root, args.dst_root, split, args.max_samples)
            all_motions.extend(motions)
        
        if 'phoenix' in datasets:
            motions = process_phoenix(args.src_root, args.dst_root, split, args.max_samples)
            all_motions.extend(motions)
    
    # Compute and save mean/std
    print(f"\n[Computing mean/std]")
    if len(all_motions) == 0:
        print("  ERROR: No motions collected!")
        return
    
    all_frames = np.concatenate(all_motions, axis=0)
    print(f"  Total sequences: {len(all_motions)}")
    print(f"  Total frames: {all_frames.shape[0]:,}")
    
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)
    std = np.clip(std, 1e-5, None)  # Prevent div by zero
    
    mean_path = os.path.join(args.dst_root, 'mean_240.pt')
    std_path = os.path.join(args.dst_root, 'std_240.pt')
    
    torch.save(torch.from_numpy(mean).float(), mean_path)
    torch.save(torch.from_numpy(std).float(), std_path)
    
    print(f"  Saved: {mean_path}")
    print(f"  Saved: {std_path}")
    
    # Statistics
    print(f"\n[Statistics]")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    parts = {'body': (0, 60), 'lhand': (60, 150), 'rhand': (150, 240)}
    for name, (s, e) in parts.items():
        print(f"    {name}: mean=[{mean[s:e].min():.3f}, {mean[s:e].max():.3f}], "
              f"std=[{std[s:e].min():.3f}, {std[s:e].max():.3f}]")
    
    # Summary
    print(f"\n[Summary]")
    print(f"  Output directory: {args.dst_root}")
    
    total_files = 0
    for dataset in ['How2Sign', 'CSL-Daily', 'Phoenix_2014T']:
        for split in splits:
            split_dir = os.path.join(args.dst_root, dataset, split)
            if os.path.exists(split_dir):
                count = len([f for f in os.listdir(split_dir) if f.endswith('.npy')])
                total_files += count
                print(f"    {dataset}/{split}: {count} files")
    
    print(f"  Total files: {total_files}")
    print("\n✅ Done!")


if __name__ == '__main__':
    main()