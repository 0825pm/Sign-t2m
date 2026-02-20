"""
preprocess_480d.py — 120D axis-angle → 480D HumanML3D-style representation

480D = joint_positions(120) + joint_velocities(120) + 6D_rotations(240)

Pipeline:
  1. pkl → 179D SMPL-X params
  2. SMPL-X FK → 3D joint positions (40 joints, centered at pelvis)
  3. axis-angle → 6D rotation (continuous, no gimbal lock)
  4. frame diff → joint velocities
  5. concatenate → 480D per frame
  6. save .npy per sample + compute mean/std

Usage:
    python preprocess_480d.py --dataset how2sign --split train
    python preprocess_480d.py --dataset how2sign --split val
    python preprocess_480d.py --dataset how2sign --split test
    python preprocess_480d.py --dataset all       # all datasets × all splits
    python preprocess_480d.py --compute_stats      # compute mean/std from saved npy
"""

import os
import sys
import gzip
import math
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import smplx
from pathlib import Path

# ─── Paths ───
H2S_ROOT     = "/home/user/Projects/research/SOKE/data/How2Sign"
CSL_ROOT     = "/home/user/Projects/research/SOKE/data/CSL-Daily"
PHOENIX_ROOT = "/home/user/Projects/research/SOKE/data/Phoenix_2014T"
SMPLX_MODEL  = "deps/smpl_models"  # relative to project root
OUTPUT_ROOT  = "/home/user/Projects/research/SOKE/data"  # output alongside existing data

# ─── Joint indices for our 44 joints in SMPL-X space ───
# Pelvis + Spine chain + upper body + hands
JOINT_SPINE = [0, 3, 6, 9]             # Pelvis, Spine1, Spine2, Spine3 (4)
JOINT_UPPER_BODY = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 10
JOINT_LHAND = list(range(25, 40))       # 15
JOINT_RHAND = list(range(40, 55))       # 15
JOINT_SELECT = JOINT_SPINE + JOINT_UPPER_BODY + JOINT_LHAND + JOINT_RHAND  # 44 joints
PELVIS_IDX = 0  # for centering
N_JOINTS = 44
FEAT_DIM = N_JOINTS * 3 * 2 + N_JOINTS * 6  # pos(132) + vel(132) + 6d(264) = 528

# ─── Default shape (betas) from SOKE ───
DEFAULT_SHAPE = np.array([
    -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
    0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
], dtype=np.float32)

# ─── Bad IDs (from existing pipeline) ───
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


# ═══════════════════════════════════════════════════════════
# 1. Rotation conversions (numpy)
# ═══════════════════════════════════════════════════════════

def axis_angle_to_matrix_np(aa):
    """axis-angle [..., 3] → rotation matrix [..., 3, 3] (Rodrigues)"""
    shape = aa.shape[:-1]
    aa_flat = aa.reshape(-1, 3)
    angle = np.linalg.norm(aa_flat, axis=-1, keepdims=True)  # [N, 1]
    axis = np.where(angle > 1e-8, aa_flat / angle, np.array([1., 0., 0.]))

    cos_a = np.cos(angle)      # [N, 1]
    sin_a = np.sin(angle)      # [N, 1]
    K = np.zeros((aa_flat.shape[0], 3, 3), dtype=aa_flat.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = np.eye(3, dtype=aa_flat.dtype)[None]  # [1, 3, 3]
    R = I + sin_a[..., None] * K + (1 - cos_a[..., None]) * (K @ K)
    return R.reshape(shape + (3, 3))


def matrix_to_6d_np(R):
    """rotation matrix [..., 3, 3] → 6D [..., 6] (first two columns)"""
    return R[..., :2, :].reshape(R.shape[:-2] + (6,))


def axis_angle_to_6d_np(aa):
    """axis-angle [..., 3] → 6D [..., 6]"""
    return matrix_to_6d_np(axis_angle_to_matrix_np(aa))


# ═══════════════════════════════════════════════════════════
# 2. SMPL-X Forward Kinematics
# ═══════════════════════════════════════════════════════════

class SMPLXForwardKinematics:
    """Efficient batch FK using SMPL-X model"""

    def __init__(self, model_path, device='cuda:0', batch_size=512):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        layer_arg = {
            'create_global_orient': False, 'create_body_pose': False,
            'create_left_hand_pose': False, 'create_right_hand_pose': False,
            'create_jaw_pose': False, 'create_leye_pose': False,
            'create_reye_pose': False, 'create_betas': False,
            'create_expression': False, 'create_transl': False,
        }
        self.model = smplx.create(
            model_path, 'smplx', gender='NEUTRAL',
            use_pca=False, use_face_contour=True, **layer_arg
        ).to(self.device).eval()

        print(f"[FK] SMPL-X loaded on {self.device}")

    @torch.no_grad()
    def forward(self, params_179):
        """
        179D SMPL-X params → 40 joint positions (centered at pelvis)

        Args:
            params_179: [T, 179] numpy array
        Returns:
            positions_40: [T, 40, 3] numpy array (pelvis-centered)
        """
        T = params_179.shape[0]
        all_positions = []

        for start in range(0, T, self.batch_size):
            end = min(start + self.batch_size, T)
            batch = torch.from_numpy(params_179[start:end]).float().to(self.device)
            B = batch.shape[0]

            root_pose = batch[:, 0:3]
            body_pose = batch[:, 3:66]
            lhand_pose = batch[:, 66:111]
            rhand_pose = batch[:, 111:156]
            jaw_pose = batch[:, 156:159]
            shape = batch[:, 159:169]
            expr = batch[:, 169:179]

            zero_pose = torch.zeros(B, 3, device=self.device)

            output = self.model(
                betas=shape, body_pose=body_pose,
                global_orient=root_pose,
                left_hand_pose=lhand_pose, right_hand_pose=rhand_pose,
                jaw_pose=jaw_pose,
                leye_pose=zero_pose, reye_pose=zero_pose,
                expression=expr,
            )

            joints = output.joints  # [B, num_joints, 3]

            # Center at pelvis
            pelvis = joints[:, PELVIS_IDX:PELVIS_IDX+1, :]  # [B, 1, 3]
            joints_centered = joints - pelvis

            # Select 40 joints
            selected = joints_centered[:, JOINT_SELECT, :]  # [B, 40, 3]
            all_positions.append(selected.cpu().numpy())

        return np.concatenate(all_positions, axis=0)  # [T, 40, 3]


# ═══════════════════════════════════════════════════════════
# 3. Convert single sample: 179D frames → 480D
# ═══════════════════════════════════════════════════════════

def convert_sample(params_179, fk_model):
    """
    Args:
        params_179: [T, 179] raw SMPL-X params
        fk_model: SMPLXForwardKinematics instance
    Returns:
        motion_528: [T, 528] = positions(132) + velocities(132) + 6D_rot(264)
    """
    T = params_179.shape[0]

    # ① FK → joint positions [T, 44, 3] → [T, 132]
    positions = fk_model.forward(params_179)               # [T, 44, 3]
    positions_flat = positions.reshape(T, -1)               # [T, 132]

    # ② Velocities [T, 132] (first frame = 0)
    velocities = np.zeros_like(positions_flat)
    velocities[1:] = positions_flat[1:] - positions_flat[:-1]

    # ③ Axis-angle → 6D rotation [T, 264]
    # 44 joints order: Pelvis, Spine1, Spine2, Spine3, Neck..R_Wrist, LHand(15), RHand(15)
    # 179D layout: root[0:3], body_pose[3:66](joints1-21), lhand[66:111], rhand[111:156]
    aa_parts = [
        params_179[:, 0:3],      # Pelvis (root_pose)
        params_179[:, 9:12],     # Spine1 (body_pose joint 3, idx=(3-1)*3+3=9)
        params_179[:, 18:21],    # Spine2 (body_pose joint 6, idx=(6-1)*3+3=18)
        params_179[:, 27:30],    # Spine3 (body_pose joint 9, idx=(9-1)*3+3=27)
        params_179[:, 36:66],    # Upper body 10 joints (body_pose joints 12-21)
        params_179[:, 66:111],   # LHand 15 joints
        params_179[:, 111:156],  # RHand 15 joints
    ]
    aa_132 = np.concatenate(aa_parts, axis=-1)             # [T, 132]
    aa_joints = aa_132.reshape(T, N_JOINTS, 3)             # [T, 44, 3]
    rot_6d = axis_angle_to_6d_np(aa_joints)                # [T, 44, 6]
    rot_6d_flat = rot_6d.reshape(T, -1)                    # [T, 264]

    # ④ Concatenate → 528D
    motion = np.concatenate([
        positions_flat,   # [T, 132]  joint positions (pelvis-centered)
        velocities,       # [T, 132]  joint velocities
        rot_6d_flat,      # [T, 264]  6D rotations
    ], axis=-1)

    return motion.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# 4. Dataset-specific loading (reuses existing pkl loading)
# ═══════════════════════════════════════════════════════════

import re

def extract_frame_num(filename):
    match = re.search(r'_(\d+)_3D\.pkl$', filename)
    if match: return int(match.group(1))
    match = re.search(r'_?(\d+)\.pkl$', filename)
    if match: return int(match.group(1))
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0


def _subsample(input_list, count):
    if count >= len(input_list): return input_list
    ss = float(len(input_list)) / count
    return [input_list[int(math.floor(i * ss))] for i in range(count)]


SOKE_KEYS = [
    'smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose',
    'smplx_rhand_pose', 'smplx_jaw_pose', 'smplx_shape', 'smplx_expr'
]
NEW_KEYS = [
    'global_orient', 'body_pose', 'left_hand_pose',
    'right_hand_pose', 'jaw_pose', 'betas', 'expression'
]


def get_pose_from_pkl(poses_dict):
    pose_values = []
    if all(k in poses_dict for k in SOKE_KEYS):
        for k in SOKE_KEYS:
            pose_values.append(np.array(poses_dict[k]).flatten())
    else:
        for new_k, soke_k in zip(NEW_KEYS, SOKE_KEYS):
            if new_k in poses_dict:
                pose_values.append(np.array(poses_dict[new_k]).flatten())
            elif soke_k in poses_dict:
                pose_values.append(np.array(poses_dict[soke_k]).flatten())
            else:
                return None
    return np.concatenate(pose_values) if pose_values else None


def load_h2s_179(name, split, fps=25):
    """Load How2Sign sample as 179D"""
    pose_dir = os.path.join(H2S_ROOT, split, 'poses', name)
    if not os.path.exists(pose_dir):
        pose_dir = os.path.join(H2S_ROOT, 'poses', name)
    if not os.path.exists(pose_dir):
        return None

    pkl_files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    if fps > 24:
        target = int(24 * len(pkl_files) / fps)
        pkl_files = _subsample(pkl_files, target)
    if len(pkl_files) < 4:
        return None

    clip = np.zeros([len(pkl_files), 179], dtype=np.float32)
    for i, f in enumerate(pkl_files):
        try:
            with open(os.path.join(pose_dir, f), 'rb') as fh:
                d = pickle.load(fh)
            pose = get_pose_from_pkl(d)
            if pose is not None and len(pose) >= 179:
                clip[i] = pose[:179]
            elif pose is not None:
                clip[i, :len(pose)] = pose
        except:
            continue
    return clip


def load_csl_179(name):
    """Load CSL-Daily sample as 179D"""
    pose_dir = os.path.join(CSL_ROOT, 'poses', name)
    if not os.path.exists(pose_dir):
        return None
    pkl_files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    if len(pkl_files) < 4:
        return None

    clip = np.zeros([len(pkl_files), 179], dtype=np.float32)
    for i, f in enumerate(pkl_files):
        try:
            with open(os.path.join(pose_dir, f), 'rb') as fh:
                d = pickle.load(fh)
            pose = get_pose_from_pkl(d)
            if pose is not None and len(pose) >= 179:
                clip[i] = pose[:179]
            elif pose is not None:
                clip[i, :len(pose)] = pose
        except:
            continue
    return clip


def load_phoenix_179(name):
    """Load Phoenix sample as 179D"""
    pose_dir = os.path.join(PHOENIX_ROOT, name)
    if not os.path.exists(pose_dir):
        return None
    pkl_files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    if len(pkl_files) < 4:
        return None

    clip = np.zeros([len(pkl_files), 179], dtype=np.float32)
    for i, f in enumerate(pkl_files):
        try:
            with open(os.path.join(pose_dir, f), 'rb') as fh:
                d = pickle.load(fh)
            pose = get_pose_from_pkl(d)
            if pose is not None and len(pose) >= 179:
                clip[i] = pose[:179]
            elif pose is not None:
                clip[i, :len(pose)] = pose
        except:
            continue
    return clip


# ═══════════════════════════════════════════════════════════
# 5. Process dataset
# ═══════════════════════════════════════════════════════════

def get_h2s_annotations(split):
    """How2Sign annotations (same as existing pipeline)"""
    csv_path = os.path.join(H2S_ROOT, split, 're_aligned',
                            f'how2sign_realigned_{split}_preprocessed_fps.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(H2S_ROOT, split, 'preprocessed_fps.csv')
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    df['DURATION'] = df['END_REALIGNED'] - df['START_REALIGNED']
    df = df[df['DURATION'] < 30].reset_index(drop=True)
    anns = []
    for i in range(len(df)):
        name = df.iloc[i]['SENTENCE_NAME']
        if name in BAD_IDS:
            continue
        anns.append({
            'name': name,
            'text': str(df.iloc[i].get('SENTENCE', '')),
            'fps': df.iloc[i].get('fps', 25),
            'split': split,
        })
    return anns


def get_csl_annotations(split):
    ann_path = os.path.join(CSL_ROOT, f'csl_clean.{split}')
    if not os.path.exists(ann_path):
        return []
    with gzip.open(ann_path, 'rb') as f:
        anns = pickle.load(f)
    for a in anns:
        a['src'] = 'csl'
    return anns


def get_phoenix_annotations(split):
    if split == 'val':
        ann_path = os.path.join(PHOENIX_ROOT, 'phoenix14t.dev')
    else:
        ann_path = os.path.join(PHOENIX_ROOT, f'phoenix14t.{split}')
    if not os.path.exists(ann_path):
        return []
    with gzip.open(ann_path, 'rb') as f:
        anns = pickle.load(f)
    for a in anns:
        a['src'] = 'phoenix'
    return anns


def process_dataset(dataset, split, fk_model):
    """Process one dataset/split → save per-sample .npy"""

    if dataset == 'how2sign':
        anns = get_h2s_annotations(split)
        out_dir = os.path.join(OUTPUT_ROOT, 'How2Sign_528d', split, 'poses')
    elif dataset == 'csl':
        anns = get_csl_annotations(split)
        out_dir = os.path.join(OUTPUT_ROOT, 'CSL-Daily_528d', 'poses')
    elif dataset == 'phoenix':
        anns = get_phoenix_annotations(split)
        out_dir = os.path.join(OUTPUT_ROOT, 'Phoenix_528d', split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Processing {dataset}/{split}: {len(anns)} samples → {out_dir}")
    print(f"{'='*60}")

    success, skip, fail = 0, 0, 0
    for ann in tqdm(anns, desc=f'{dataset}/{split}'):
        name = ann['name']
        save_name = name.split('/')[-1] if '/' in name else name
        npy_path = os.path.join(out_dir, f'{save_name}.npy')

        # Skip if already exists
        if os.path.exists(npy_path):
            skip += 1
            continue

        # Load 179D
        try:
            if dataset == 'how2sign':
                params = load_h2s_179(name, split, fps=ann.get('fps', 25))
            elif dataset == 'csl':
                params = load_csl_179(name)
            elif dataset == 'phoenix':
                params = load_phoenix_179(name)
            else:
                params = None

            if params is None or len(params) < 4:
                fail += 1
                continue

            # Convert 179D → 528D
            motion = convert_sample(params, fk_model)
            np.save(npy_path, motion)
            success += 1

        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  ✗ {name}: {e}")

    print(f"  Done: {success} saved, {skip} skipped, {fail} failed")
    return success


# ═══════════════════════════════════════════════════════════
# 6. Compute normalization statistics
# ═══════════════════════════════════════════════════════════

def compute_stats(dataset):
    """Compute mean/std from train split .npy files"""

    if dataset == 'how2sign':
        npy_dir = os.path.join(OUTPUT_ROOT, 'How2Sign_528d', 'train', 'poses')
        stat_dir = os.path.join(OUTPUT_ROOT, 'How2Sign_528d')
    elif dataset == 'csl':
        npy_dir = os.path.join(OUTPUT_ROOT, 'CSL-Daily_528d', 'poses')
        stat_dir = os.path.join(OUTPUT_ROOT, 'CSL-Daily_528d')
    elif dataset == 'phoenix':
        npy_dir = os.path.join(OUTPUT_ROOT, 'Phoenix_528d', 'train')
        stat_dir = os.path.join(OUTPUT_ROOT, 'Phoenix_528d')
    else:
        raise ValueError(dataset)

    print(f"\nComputing stats for {dataset} from {npy_dir}")
    if not os.path.exists(npy_dir):
        print(f"  ✗ Directory not found")
        return

    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    print(f"  Found {len(npy_files)} files")

    # Two-pass: first pass for mean, second for std
    total_frames = 0
    running_sum = np.zeros(FEAT_DIM, dtype=np.float64)

    for f in tqdm(npy_files, desc='Pass 1 (mean)'):
        data = np.load(os.path.join(npy_dir, f))  # [T, FEAT_DIM]
        running_sum += data.sum(axis=0).astype(np.float64)
        total_frames += data.shape[0]

    mean = (running_sum / total_frames).astype(np.float32)

    running_sq = np.zeros(FEAT_DIM, dtype=np.float64)
    for f in tqdm(npy_files, desc='Pass 2 (std)'):
        data = np.load(os.path.join(npy_dir, f))
        running_sq += ((data - mean) ** 2).sum(axis=0).astype(np.float64)

    std = np.sqrt(running_sq / total_frames).astype(np.float32)
    std = np.clip(std, 1e-6, None)  # avoid division by zero

    # Save
    mean_path = os.path.join(stat_dir, 'mean_528.npy')
    std_path = os.path.join(stat_dir, 'std_528.npy')
    np.save(mean_path, mean)
    np.save(std_path, std)
    print(f"  ✅ Saved: {mean_path}")
    print(f"  ✅ Saved: {std_path}")

    # Print summary
    print(f"\n  528D stats ({total_frames} frames):")
    print(f"  {'Part':<15s} {'Dims':<10s} {'Mean range':<25s} {'Std range':<25s}")
    print(f"  {'-'*75}")
    parts = [('positions', 0, N_JOINTS*3), ('velocities', N_JOINTS*3, N_JOINTS*6), ('6D_rot', N_JOINTS*6, FEAT_DIM)]
    for name, s, e in parts:
        m, st = mean[s:e], std[s:e]
        print(f"  {name:<15s} [{s}:{e}]    [{m.min():.4f}, {m.max():.4f}]    [{st.min():.4f}, {st.max():.4f}]")

    # Also save as torch tensors for training
    torch.save(torch.from_numpy(mean), os.path.join(stat_dir, 'mean_528.pt'))
    torch.save(torch.from_numpy(std), os.path.join(stat_dir, 'std_528.pt'))
    print(f"  ✅ Also saved .pt versions")


# ═══════════════════════════════════════════════════════════
# 7. Unified stats (CSL mean/std used for all, like current)
# ═══════════════════════════════════════════════════════════

def compute_unified_stats():
    """Compute unified mean/std from all datasets' train split"""

    dirs = []
    for d, sub in [('How2Sign_528d', 'train/poses'), ('CSL-Daily_528d', 'poses'), ('Phoenix_528d', 'train')]:
        p = os.path.join(OUTPUT_ROOT, d, sub)
        if os.path.exists(p):
            dirs.append(p)

    if not dirs:
        print("No 528d data found")
        return

    print(f"\nComputing unified stats from {len(dirs)} directories")

    total_frames = 0
    running_sum = np.zeros(FEAT_DIM, dtype=np.float64)

    for d in dirs:
        files = [f for f in os.listdir(d) if f.endswith('.npy')]
        print(f"  {d}: {len(files)} files")
        for f in tqdm(files, desc=os.path.basename(os.path.dirname(d)), leave=False):
            data = np.load(os.path.join(d, f))
            running_sum += data.sum(axis=0).astype(np.float64)
            total_frames += data.shape[0]

    mean = (running_sum / total_frames).astype(np.float32)

    running_sq = np.zeros(FEAT_DIM, dtype=np.float64)
    for d in dirs:
        files = [f for f in os.listdir(d) if f.endswith('.npy')]
        for f in tqdm(files, desc='std', leave=False):
            data = np.load(os.path.join(d, f))
            running_sq += ((data - mean) ** 2).sum(axis=0).astype(np.float64)

    std = np.sqrt(running_sq / total_frames).astype(np.float32)
    std = np.clip(std, 1e-6, None)

    stat_dir = os.path.join(OUTPUT_ROOT, 'unified_528d')
    os.makedirs(stat_dir, exist_ok=True)
    np.save(os.path.join(stat_dir, 'mean_528.npy'), mean)
    np.save(os.path.join(stat_dir, 'std_528.npy'), std)
    torch.save(torch.from_numpy(mean), os.path.join(stat_dir, 'mean_528.pt'))
    torch.save(torch.from_numpy(std), os.path.join(stat_dir, 'std_528.pt'))
    print(f"  ✅ Saved unified stats to {stat_dir} ({total_frames} total frames)")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=['how2sign', 'csl', 'phoenix', 'all'])
    parser.add_argument('--split', default='all', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--compute_stats', action='store_true', help='Only compute mean/std')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--smplx_path', default=SMPLX_MODEL)
    args = parser.parse_args()

    if args.compute_stats:
        for ds in (['how2sign', 'csl', 'phoenix'] if args.dataset == 'all' else [args.dataset]):
            compute_stats(ds)
        compute_unified_stats()
        return

    # Initialize FK model
    fk_model = SMPLXForwardKinematics(args.smplx_path, args.device, args.batch_size)

    datasets = ['how2sign', 'csl', 'phoenix'] if args.dataset == 'all' else [args.dataset]
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    for ds in datasets:
        for sp in splits:
            process_dataset(ds, sp, fk_model)

    # After processing, compute stats
    print("\n\n" + "="*60)
    print("  Computing normalization statistics")
    print("="*60)
    for ds in datasets:
        compute_stats(ds)
    if len(datasets) > 1:
        compute_unified_stats()

    print("\n✅ All done!")
    print("\nNext steps:")
    print("  1. Update configs to use 528D data")
    print("  2. Update model motion_dim: 120 → 528")
    print("  3. Train: python src/train.py data.dataset_name=how2sign trainer.max_epochs=1000")


if __name__ == "__main__":
    main()