"""
SOKE-style data loading functions for Sign Language
Supports both SOKE keys (smplx_*) and new keys (global_orient, body_pose, etc.)
"""
import os
import re
import math
import pickle
import numpy as np
from bisect import bisect_left, bisect_right


# SOKE 원본 키 (179 dims)
SOKE_KEYS = [
    'smplx_root_pose',    # 3
    'smplx_body_pose',    # 63
    'smplx_lhand_pose',   # 45
    'smplx_rhand_pose',   # 45
    'smplx_jaw_pose',     # 3
    'smplx_shape',        # 10
    'smplx_expr'          # 10
]

# 새로운 키 형식
NEW_KEYS = [
    'global_orient',      # 3
    'body_pose',          # 63
    'left_hand_pose',     # 45
    'right_hand_pose',    # 45
    'jaw_pose',           # 3
    'betas',              # 10
    'expression'          # 10
]


def extract_frame_num(filename):
    """파일명에서 프레임 번호 추출 (숫자 기반 정렬용)
    
    패턴 예시:
    - name_0_3D.pkl -> 0
    - name_100_3D.pkl -> 100
    - frame_0001.pkl -> 1
    - 0001.pkl -> 1
    """
    # _숫자_3D.pkl 패턴 (How2Sign, CSL)
    match = re.search(r'_(\d+)_3D\.pkl$', filename)
    if match:
        return int(match.group(1))
    
    # _숫자.pkl 또는 숫자.pkl 패턴
    match = re.search(r'_?(\d+)\.pkl$', filename)
    if match:
        return int(match.group(1))
    
    # 파일명 어디든 숫자가 있으면 추출
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])  # 마지막 숫자 사용
    
    return 0


def get_pose_from_pkl(poses_dict):
    """pkl 파일에서 pose 데이터 추출"""
    pose_values = []
    
    all_soke_keys_found = all(key in poses_dict for key in SOKE_KEYS)
    
    if all_soke_keys_found:
        for key in SOKE_KEYS:
            val = np.array(poses_dict[key]).flatten()
            pose_values.append(val)
    else:
        key_order = [
            ('global_orient', 'smplx_root_pose'),
            ('body_pose', 'smplx_body_pose'),
            ('left_hand_pose', 'smplx_lhand_pose'),
            ('right_hand_pose', 'smplx_rhand_pose'),
            ('jaw_pose', 'smplx_jaw_pose'),
            ('betas', 'smplx_shape'),
            ('expression', 'smplx_expr')
        ]
        
        for new_key, soke_key in key_order:
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


def convert_179_to_120(clip_poses):
    """179-dim → 120-dim (upper_body + lhand + rhand only)
    
    179-dim에서 직접 추출:
    - [36:66]   upper_body (10 joints × 3 = 30)
    - [66:111]  lhand (15 joints × 3 = 45)
    - [111:156] rhand (15 joints × 3 = 45)
    
    Total: 30 + 45 + 45 = 120
    """
    return clip_poses[:, 36:156]


def load_h2s_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """Load How2Sign sample"""
    clip_text = ann['text']
    name = ann['name']
    split = ann.get('split', 'train')
    
    pose_dir = os.path.join(data_dir, split, 'poses', name)
    if not os.path.exists(pose_dir):
        pose_dir = os.path.join(data_dir, 'poses', name)
    if not os.path.exists(pose_dir):
        return None, None, None, None
    
    # 숫자 기반 정렬 (중요!)
    pkl_files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    if len(pkl_files) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(pkl_files), 179], dtype=np.float32)
    
    if need_pose:
        for frame_id, frame in enumerate(pkl_files):
            frame_path = os.path.join(pose_dir, frame)
            try:
                with open(frame_path, 'rb') as f:
                    poses_dict = pickle.load(f)
                pose = get_pose_from_pkl(poses_dict)
                if pose is not None and len(pose) >= 179:
                    clip_poses[frame_id] = pose[:179]
                elif pose is not None:
                    clip_poses[frame_id, :len(pose)] = pose
            except:
                continue
        
        clip_poses = convert_179_to_120(clip_poses)
    else:
        clip_poses = np.zeros([len(pkl_files), 120], dtype=np.float32)
    
    return clip_poses.astype(np.float32), clip_text, name, None


def load_csl_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """Load CSL-Daily sample"""
    clip_text = ann['text']
    name = ann['name']
    
    pose_dir = os.path.join(data_dir, 'poses', name)
    if not os.path.exists(pose_dir):
        return None, None, None, None
    
    # 숫자 기반 정렬 (중요!)
    frame_list = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    if len(frame_list) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179], dtype=np.float32)
    
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(pose_dir, frame)
            try:
                with open(frame_path, 'rb') as f:
                    poses_dict = pickle.load(f)
                pose = get_pose_from_pkl(poses_dict)
                if pose is not None and len(pose) >= 179:
                    clip_poses[frame_id] = pose[:179]
                elif pose is not None:
                    clip_poses[frame_id, :len(pose)] = pose
            except:
                continue
        
        clip_poses = convert_179_to_120(clip_poses)
    else:
        clip_poses = np.zeros([len(frame_list), 120], dtype=np.float32)
    
    return clip_poses.astype(np.float32), clip_text, name, None


def load_phoenix_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """Load Phoenix-2014T sample"""
    clip_text = ann['text']
    name = ann['name']
    
    pose_dir = os.path.join(data_dir, name)
    if not os.path.exists(pose_dir):
        return None, None, None, None
    
    # 숫자 기반 정렬 (중요!)
    frame_list = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    if len(frame_list) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179], dtype=np.float32)
    
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(pose_dir, frame)
            try:
                with open(frame_path, 'rb') as f:
                    poses_dict = pickle.load(f)
                pose = get_pose_from_pkl(poses_dict)
                if pose is not None and len(pose) >= 179:
                    clip_poses[frame_id] = pose[:179]
                elif pose is not None:
                    clip_poses[frame_id, :len(pose)] = pose
            except:
                continue
        
        clip_poses = convert_179_to_120(clip_poses)
    else:
        clip_poses = np.zeros([len(frame_list), 120], dtype=np.float32)
    
    return clip_poses.astype(np.float32), clip_text, name, None