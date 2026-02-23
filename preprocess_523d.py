"""
preprocess_523d.py — 179D SMPLX → 523D HumanML3D-style sign language representation

HumanML3D 구조를 수어에 맞게 적용:
  Body (14 joints, pelvis-relative) = 163D
  Left Hand (15 joints, wrist-relative) = 180D
  Right Hand (15 joints, wrist-relative) = 180D

Layout:
  ── Body (163D) ──────────────────────────────────
  [0]       root_rot_velocity        (1)   pelvis Y-axis 회전 속도
  [1:3]     root_linear_velocity     (2)   pelvis XZ 속도
  [3]       root_y                   (1)   pelvis Y 높이
  [4:43]    body_ric                 (39)  joints 1-13 pelvis-relative positions
  [43:121]  body_rot                 (78)  joints 1-13 6D rotations
  [121:163] body_vel                 (42)  joints 0-13 velocity
  ── Left Hand (180D) ─────────────────────────────
  [163:208] lhand_ric                (45)  15 joints lwrist-relative positions
  [208:298] lhand_rot                (90)  15 joints 6D rotations
  [298:343] lhand_vel                (45)  15 joints velocity (wrist-relative)
  ── Right Hand (180D) ────────────────────────────
  [343:388] rhand_ric                (45)  15 joints rwrist-relative positions
  [388:478] rhand_rot                (90)  15 joints 6D rotations
  [478:523] rhand_vel                (45)  15 joints velocity (wrist-relative)

Pipeline:
  1. pkl → 179D SMPL-X params
  2. SMPL-X FK → 44-joint positions (pelvis-centered)
  3. Body: pelvis-relative positions + root motion extraction
  4. Hands: wrist-relative positions
  5. axis-angle → 6D rotation
  6. frame diff → velocities (per-part)
  7. concatenate → 523D per frame
  8. save .npy per sample + compute mean/std

Usage:
    python preprocess_523d.py --dataset how2sign --split train
    python preprocess_523d.py --dataset how2sign --split val
    python preprocess_523d.py --dataset all --split all
    python preprocess_523d.py --compute_stats
"""

import os
import sys
import gzip
import math
import pickle
import argparse
import re
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import smplx
from pathlib import Path

# ─── Paths ───
H2S_ROOT     = "/home/user/Projects/research/SOKE/data/How2Sign"
CSL_ROOT     = "/home/user/Projects/research/SOKE/data/CSL-Daily"
PHOENIX_ROOT = "/home/user/Projects/research/SOKE/data/Phoenix_2014T"
SMPLX_MODEL  = "deps/smpl_models"
OUTPUT_ROOT  = "/home/user/Projects/research/SOKE/data"

# ─── 44 joints in SMPL-X space ───
JOINT_SPINE = [0, 3, 6, 9]                                    # 4
JOINT_UPPER_BODY = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 10
JOINT_LHAND = list(range(25, 40))                              # 15
JOINT_RHAND = list(range(40, 55))                              # 15
JOINT_SELECT = JOINT_SPINE + JOINT_UPPER_BODY + JOINT_LHAND + JOINT_RHAND  # 44

# In our 44-joint order:
#   0-3:   spine (pelvis, spine1, spine2, spine3)
#   4-13:  upper body (neck, l_collar, r_collar, head, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist)
#   14-28: left hand (15 joints)
#   29-43: right hand (15 joints)
IDX_BODY = slice(0, 14)     # 14 joints
IDX_LWRIST = 12              # left wrist index in 44-joint
IDX_RWRIST = 13              # right wrist index in 44-joint
IDX_LHAND = slice(14, 29)   # 15 joints
IDX_RHAND = slice(29, 44)   # 15 joints

PELVIS_IDX = 0
N_JOINTS = 44
N_BODY = 14
N_HAND = 15

# 523D layout dimensions
DIM_BODY = 1 + 2 + 1 + 13*3 + 13*6 + 14*3   # = 163
DIM_HAND = 15*3 + 15*6 + 15*3                 # = 180
FEAT_DIM = DIM_BODY + DIM_HAND * 2            # = 523

# ─── Default shape (betas) from SOKE ───
DEFAULT_SHAPE = np.array([
    -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
    0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
], dtype=np.float32)

# ─── Bad IDs ───
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
# 1. Rotation conversions
# ═══════════════════════════════════════════════════════════

def axis_angle_to_matrix_np(aa):
    """axis-angle [..., 3] → rotation matrix [..., 3, 3] (Rodrigues)"""
    shape = aa.shape[:-1]
    aa_flat = aa.reshape(-1, 3)
    angle = np.linalg.norm(aa_flat, axis=-1, keepdims=True)
    safe_angle = np.where(angle > 1e-8, angle, 1.0)
    axis = np.where(angle > 1e-8, aa_flat / safe_angle, np.array([1., 0., 0.]))

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    K = np.zeros((aa_flat.shape[0], 3, 3), dtype=aa_flat.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = np.eye(3, dtype=aa_flat.dtype)[None]
    R = I + sin_a[..., None] * K + (1 - cos_a[..., None]) * (K @ K)
    return R.reshape(shape + (3, 3))


def matrix_to_6d_np(R):
    """rotation matrix [..., 3, 3] → 6D [..., 6]"""
    return R[..., :2, :].reshape(R.shape[:-2] + (6,))


def axis_angle_to_6d_np(aa):
    """axis-angle [..., 3] → 6D [..., 6]"""
    return matrix_to_6d_np(axis_angle_to_matrix_np(aa))


# ═══════════════════════════════════════════════════════════
# 2. SMPL-X Forward Kinematics
# ═══════════════════════════════════════════════════════════

class SMPLXForwardKinematics:
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
        """179D → 44-joint positions [T, 44, 3] (pelvis-centered)"""
        T = params_179.shape[0]
        all_positions = []
        for start in range(0, T, self.batch_size):
            end = min(start + self.batch_size, T)
            batch = torch.from_numpy(params_179[start:end]).float().to(self.device)
            B = batch.shape[0]
            zero_pose = torch.zeros(B, 3, device=self.device)
            output = self.model(
                betas=batch[:, 159:169], body_pose=batch[:, 3:66],
                global_orient=batch[:, 0:3],
                left_hand_pose=batch[:, 66:111], right_hand_pose=batch[:, 111:156],
                jaw_pose=batch[:, 156:159],
                leye_pose=zero_pose, reye_pose=zero_pose,
                expression=batch[:, 169:179],
            )
            joints = output.joints
            pelvis = joints[:, PELVIS_IDX:PELVIS_IDX+1, :]
            joints_centered = joints - pelvis
            selected = joints_centered[:, JOINT_SELECT, :]
            all_positions.append(selected.cpu().numpy())
        return np.concatenate(all_positions, axis=0)


# ═══════════════════════════════════════════════════════════
# 3. Convert single sample: 179D → 523D
# ═══════════════════════════════════════════════════════════

def convert_sample(params_179, fk_model):
    """
    179D SMPLX params → 523D HumanML3D-style representation

    Returns: [T, 523] numpy array
    """
    T = params_179.shape[0]

    # ── FK → positions [T, 44, 3] pelvis-centered ──
    positions = fk_model.forward(params_179)  # [T, 44, 3]

    # ── Separate parts ──
    body_pos = positions[:, IDX_BODY, :]     # [T, 14, 3]
    lhand_pos = positions[:, IDX_LHAND, :]   # [T, 15, 3]
    rhand_pos = positions[:, IDX_RHAND, :]   # [T, 15, 3]

    # ── Hand → wrist-relative ──
    lwrist = positions[:, IDX_LWRIST:IDX_LWRIST+1, :]  # [T, 1, 3]
    rwrist = positions[:, IDX_RWRIST:IDX_RWRIST+1, :]  # [T, 1, 3]
    lhand_ric = lhand_pos - lwrist   # [T, 15, 3]
    rhand_ric = rhand_pos - rwrist   # [T, 15, 3]

    # ── Body ric (joints 1-13, exclude pelvis) ──
    body_ric = body_pos[:, 1:, :]    # [T, 13, 3]

    # ── Root motion (from global_orient) ──
    global_orient = params_179[:, 0:3]                    # [T, 3] axis-angle
    R = axis_angle_to_matrix_np(global_orient)            # [T, 3, 3]
    y_angle = np.arctan2(R[:, 0, 2], R[:, 0, 0])         # [T]

    root_rot_vel = np.zeros((T, 1), dtype=np.float32)
    if T > 1:
        root_rot_vel[1:, 0] = y_angle[1:] - y_angle[:-1]

    root_linear_vel = np.zeros((T, 2), dtype=np.float32)  # XZ velocity (≈0 for sign)
    root_y = np.zeros((T, 1), dtype=np.float32)           # height (≈0 since centered)

    # ── Velocities (per-part frame diff) ──
    body_pos_flat = body_pos.reshape(T, -1)      # [T, 42]
    lhand_ric_flat = lhand_ric.reshape(T, -1)    # [T, 45]
    rhand_ric_flat = rhand_ric.reshape(T, -1)    # [T, 45]

    body_vel = np.zeros_like(body_pos_flat)
    lhand_vel = np.zeros_like(lhand_ric_flat)
    rhand_vel = np.zeros_like(rhand_ric_flat)
    if T > 1:
        body_vel[1:] = body_pos_flat[1:] - body_pos_flat[:-1]
        lhand_vel[1:] = lhand_ric_flat[1:] - lhand_ric_flat[:-1]
        rhand_vel[1:] = rhand_ric_flat[1:] - rhand_ric_flat[:-1]

    # ── 6D Rotations (per-part) ──
    aa_parts = [
        params_179[:, 0:3],       # Pelvis (root)
        params_179[:, 9:12],      # Spine1
        params_179[:, 18:21],     # Spine2
        params_179[:, 27:30],     # Spine3
        params_179[:, 36:66],     # Upper body 10 joints (12-21)
        params_179[:, 66:111],    # LHand 15 joints
        params_179[:, 111:156],   # RHand 15 joints
    ]
    aa_132 = np.concatenate(aa_parts, axis=-1)     # [T, 132]
    aa_joints = aa_132.reshape(T, N_JOINTS, 3)     # [T, 44, 3]
    rot_6d = axis_angle_to_6d_np(aa_joints)        # [T, 44, 6]

    # Exclude pelvis rotation (captured in root_rot_vel)
    body_rot = rot_6d[:, 1:14, :].reshape(T, -1)   # [T, 78]  (13 joints)
    lhand_rot = rot_6d[:, 14:29, :].reshape(T, -1)  # [T, 90]  (15 joints)
    rhand_rot = rot_6d[:, 29:44, :].reshape(T, -1)  # [T, 90]  (15 joints)

    # ── Concatenate → 523D ──
    motion = np.concatenate([
        # Body (163D)
        root_rot_vel,                    # [T, 1]
        root_linear_vel,                 # [T, 2]
        root_y,                          # [T, 1]
        body_ric.reshape(T, -1),         # [T, 39]
        body_rot,                        # [T, 78]
        body_vel,                        # [T, 42]
        # Left Hand (180D)
        lhand_ric_flat,                  # [T, 45]
        lhand_rot,                       # [T, 90]
        lhand_vel,                       # [T, 45]
        # Right Hand (180D)
        rhand_ric_flat,                  # [T, 45]
        rhand_rot,                       # [T, 90]
        rhand_vel,                       # [T, 45]
    ], axis=-1)

    assert motion.shape[-1] == FEAT_DIM, f"Expected {FEAT_DIM}D, got {motion.shape[-1]}"
    return motion.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# 4. Dataset-specific loading (528d와 동일)
# ═══════════════════════════════════════════════════════════

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
# 5. Annotations
# ═══════════════════════════════════════════════════════════

def get_h2s_annotations(split):
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


# ═══════════════════════════════════════════════════════════
# 6. Process dataset
# ═══════════════════════════════════════════════════════════

def process_dataset(dataset, split, fk_model):
    if dataset == 'how2sign':
        anns = get_h2s_annotations(split)
        out_dir = os.path.join(OUTPUT_ROOT, 'How2Sign_523d', split, 'poses')
    elif dataset == 'csl':
        anns = get_csl_annotations(split)
        out_dir = os.path.join(OUTPUT_ROOT, 'CSL-Daily_523d', 'poses')
    elif dataset == 'phoenix':
        anns = get_phoenix_annotations(split)
        out_dir = os.path.join(OUTPUT_ROOT, 'Phoenix_523d', split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Processing {dataset}/{split}: {len(anns)} samples → {out_dir}")
    print(f"  Output: {FEAT_DIM}D (body={DIM_BODY} + lhand={DIM_HAND} + rhand={DIM_HAND})")
    print(f"{'='*60}")

    success, skip, fail = 0, 0, 0
    for ann in tqdm(anns, desc=f'{dataset}/{split}'):
        name = ann['name']
        save_name = name.split('/')[-1] if '/' in name else name
        npy_path = os.path.join(out_dir, f'{save_name}.npy')

        if os.path.exists(npy_path):
            skip += 1
            continue

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
# 7. Compute normalization statistics
# ═══════════════════════════════════════════════════════════

def compute_stats(dataset):
    if dataset == 'how2sign':
        npy_dir = os.path.join(OUTPUT_ROOT, 'How2Sign_523d', 'train', 'poses')
        stat_dir = os.path.join(OUTPUT_ROOT, 'How2Sign_523d')
    elif dataset == 'csl':
        npy_dir = os.path.join(OUTPUT_ROOT, 'CSL-Daily_523d', 'poses')
        stat_dir = os.path.join(OUTPUT_ROOT, 'CSL-Daily_523d')
    elif dataset == 'phoenix':
        npy_dir = os.path.join(OUTPUT_ROOT, 'Phoenix_523d', 'train')
        stat_dir = os.path.join(OUTPUT_ROOT, 'Phoenix_523d')
    else:
        raise ValueError(dataset)

    print(f"\nComputing stats for {dataset} from {npy_dir}")
    if not os.path.exists(npy_dir):
        print(f"  ✗ Directory not found")
        return

    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    print(f"  Found {len(npy_files)} files")

    total_frames = 0
    running_sum = np.zeros(FEAT_DIM, dtype=np.float64)
    for f in tqdm(npy_files, desc='Pass 1 (mean)'):
        data = np.load(os.path.join(npy_dir, f))
        running_sum += data.sum(axis=0).astype(np.float64)
        total_frames += data.shape[0]

    mean = (running_sum / total_frames).astype(np.float32)

    running_sq = np.zeros(FEAT_DIM, dtype=np.float64)
    for f in tqdm(npy_files, desc='Pass 2 (std)'):
        data = np.load(os.path.join(npy_dir, f))
        running_sq += ((data - mean) ** 2).sum(axis=0).astype(np.float64)

    std = np.sqrt(running_sq / total_frames).astype(np.float32)
    std = np.clip(std, 1e-6, None)

    os.makedirs(stat_dir, exist_ok=True)

    # Save both numpy and torch
    np.save(os.path.join(stat_dir, 'mean_523.npy'), mean)
    np.save(os.path.join(stat_dir, 'std_523.npy'), std)
    torch.save(torch.from_numpy(mean), os.path.join(stat_dir, 'mean_523.pt'))
    torch.save(torch.from_numpy(std), os.path.join(stat_dir, 'std_523.pt'))

    # Print summary by part
    print(f"\n  523D stats ({total_frames} frames):")
    print(f"  {'Part':<20s} {'Dims':<12s} {'Mean range':<28s} {'Std range':<28s}")
    print(f"  {'-'*88}")
    parts = [
        ('root_rot_vel',     0,   1),
        ('root_linear_vel',  1,   3),
        ('root_y',           3,   4),
        ('body_ric',         4,   43),
        ('body_rot',         43,  121),
        ('body_vel',         121, 163),
        ('lhand_ric',        163, 208),
        ('lhand_rot',        208, 298),
        ('lhand_vel',        298, 343),
        ('rhand_ric',        343, 388),
        ('rhand_rot',        388, 478),
        ('rhand_vel',        478, 523),
    ]
    for name, s, e in parts:
        m, st = mean[s:e], std[s:e]
        print(f"  {name:<20s} [{s:3d}:{e:3d}]  [{m.min():+.5f}, {m.max():+.5f}]  [{st.min():.5f}, {st.max():.5f}]")

    print(f"\n  ✅ Saved to {stat_dir}")


def compute_unified_stats():
    dirs = []
    for d, sub in [('How2Sign_523d', 'train/poses'), ('CSL-Daily_523d', 'poses'), ('Phoenix_523d', 'train')]:
        p = os.path.join(OUTPUT_ROOT, d, sub)
        if os.path.exists(p):
            dirs.append(p)
    if not dirs:
        print("No 523d data found")
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

    stat_dir = os.path.join(OUTPUT_ROOT, 'unified_523d')
    os.makedirs(stat_dir, exist_ok=True)
    np.save(os.path.join(stat_dir, 'mean_523.npy'), mean)
    np.save(os.path.join(stat_dir, 'std_523.npy'), std)
    torch.save(torch.from_numpy(mean), os.path.join(stat_dir, 'mean_523.pt'))
    torch.save(torch.from_numpy(std), os.path.join(stat_dir, 'std_523.pt'))
    print(f"  ✅ Unified stats: {stat_dir} ({total_frames} frames)")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=['how2sign', 'csl', 'phoenix', 'all'])
    parser.add_argument('--split', default='all', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--compute_stats', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--smplx_path', default=SMPLX_MODEL)
    args = parser.parse_args()

    if args.compute_stats:
        for ds in (['how2sign', 'csl', 'phoenix'] if args.dataset == 'all' else [args.dataset]):
            compute_stats(ds)
        compute_unified_stats()
        return

    fk_model = SMPLXForwardKinematics(args.smplx_path, args.device, args.batch_size)

    datasets = ['how2sign', 'csl', 'phoenix'] if args.dataset == 'all' else [args.dataset]
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    for ds in datasets:
        for sp in splits:
            process_dataset(ds, sp, fk_model)

    print("\n\n" + "="*60)
    print("  Computing normalization statistics")
    print("="*60)
    for ds in datasets:
        compute_stats(ds)
    if len(datasets) > 1:
        compute_unified_stats()

    print("\n✅ All done!")
    print(f"\n523D layout:")
    print(f"  Body  [0:163]   = root_motion(4) + ric(39) + rot(78) + vel(42)")
    print(f"  LHand [163:343] = ric(45) + rot(90) + vel(45)  [wrist-relative]")
    print(f"  RHand [343:523] = ric(45) + rot(90) + vel(45)  [wrist-relative]")


if __name__ == "__main__":
    main()
