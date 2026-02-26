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


def _subsample(input_list, count):
    """Uniform subsample list to target count (from salad)"""
    if count >= len(input_list):
        return input_list
    ss = float(len(input_list)) / count
    return [input_list[int(math.floor(i * ss))] for i in range(count)]


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


def convert_179_to_133(clip_poses):
    """179-dim → 133-dim (SOKE 133D: axis-angle format)

    179-dim layout:
      [0:3]    root_pose
      [3:66]   body_pose (21j × 3)
      [66:111] lhand_pose (15j × 3)
      [111:156] rhand_pose (15j × 3)
      [156:159] jaw_pose
      [159:169] shape (betas)
      [169:179] expression

    133D = upper_body(30) + lhand(45) + rhand(45) + jaw(3) + expr(10)
      [36:156]  → 120D (same as convert_179_to_120)
      [156:159] → jaw 3D
      [169:179] → expr 10D
    """
    body_hand = clip_poses[:, 36:156]   # 120D
    jaw       = clip_poses[:, 156:159]  # 3D
    expr      = clip_poses[:, 169:179]  # 10D
    return np.concatenate([body_hand, jaw, expr], axis=-1)  # 133D


def load_h2s_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """Load How2Sign sample
    
    FPS normalization: fps > 24 → subsample to 24fps (from salad)
    """
    clip_text = ann['text']
    name = ann['name']
    split = ann.get('split', 'train')
    fps = ann.get('fps', 25)
    
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
    
    # FPS 정규화: fps > 24이면 24fps로 서브샘플링
    if fps > 24:
        target_count = int(24 * len(pkl_files) / fps)
        pkl_files = _subsample(pkl_files, target_count)
    
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


def load_h2s_sample_133(ann, data_dir, **kwargs):
    """How2Sign 133D axis-angle (SOKE format)"""
    clip_poses, text, name, _ = load_h2s_sample(ann, data_dir)
    if clip_poses is None:
        return None, None, None, None
    # load_h2s_sample returns 120D; reload raw and convert to 133D
    clip_text = ann['text']
    name = ann['name']
    split = ann.get('split', 'train')
    fps = ann.get('fps', 25)

    pose_dir = os.path.join(data_dir, split, 'poses', name)
    if not os.path.exists(pose_dir):
        pose_dir = os.path.join(data_dir, 'poses', name)
    if not os.path.exists(pose_dir):
        return None, None, None, None

    pkl_files = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    if fps > 24:
        target_count = int(24 * len(pkl_files) / fps)
        pkl_files = _subsample(pkl_files, target_count)
    if len(pkl_files) < 4:
        return None, None, None, None

    raw = np.zeros([len(pkl_files), 179], dtype=np.float32)
    for i, frame in enumerate(pkl_files):
        try:
            with open(os.path.join(pose_dir, frame), 'rb') as f:
                d = pickle.load(f)
            pose = get_pose_from_pkl(d)
            if pose is not None and len(pose) >= 179:
                raw[i] = pose[:179]
            elif pose is not None:
                raw[i, :len(pose)] = pose
        except:
            continue

    return convert_179_to_133(raw).astype(np.float32), clip_text, name, None


def load_csl_sample_133(ann, data_dir, **kwargs):
    """CSL-Daily 133D axis-angle (SOKE format)"""
    clip_text = ann['text']
    name = ann['name']

    pose_dir = os.path.join(data_dir, 'poses', name)
    if not os.path.exists(pose_dir):
        return None, None, None, None

    frame_list = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    if len(frame_list) < 4:
        return None, None, None, None

    raw = np.zeros([len(frame_list), 179], dtype=np.float32)
    for i, frame in enumerate(frame_list):
        try:
            with open(os.path.join(pose_dir, frame), 'rb') as f:
                d = pickle.load(f)
            pose = get_pose_from_pkl(d)
            if pose is not None and len(pose) >= 179:
                raw[i] = pose[:179]
            elif pose is not None:
                raw[i, :len(pose)] = pose
        except:
            continue

    return convert_179_to_133(raw).astype(np.float32), clip_text, name, None


def load_phoenix_sample_133(ann, data_dir, **kwargs):
    """Phoenix-2014T 133D axis-angle (SOKE format)"""
    clip_text = ann['text']
    name = ann['name']

    pose_dir = os.path.join(data_dir, name)
    if not os.path.exists(pose_dir):
        return None, None, None, None

    frame_list = sorted(
        [f for f in os.listdir(pose_dir) if f.endswith('.pkl')],
        key=extract_frame_num
    )
    if len(frame_list) < 4:
        return None, None, None, None

    raw = np.zeros([len(frame_list), 179], dtype=np.float32)
    for i, frame in enumerate(frame_list):
        try:
            with open(os.path.join(pose_dir, frame), 'rb') as f:
                d = pickle.load(f)
            pose = get_pose_from_pkl(d)
            if pose is not None and len(pose) >= 179:
                raw[i] = pose[:179]
            elif pose is not None:
                raw[i, :len(pose)] = pose
        except:
            continue

    return convert_179_to_133(raw).astype(np.float32), clip_text, name, None


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


# ============================================================
# 6D npy 로딩 함수 (Step 7)
# preprocess_6d.py로 생성된 240-dim npy를 직접 로드
# pkl 파싱 불필요 → I/O 대폭 개선
# ============================================================



def load_phoenix_sample_6d(ann, data6d_dir, split):
    """Load Phoenix-2014T 240-dim 6D npy
    
    Note: Phoenix name에 'train/video_name' 형태가 있을 수 있음
          preprocess_6d.py에서 split prefix 제거 후 저장했으므로 여기서도 제거
    """
    name = ann['name']
    text = ann.get('text', '')
    
    # split prefix 제거 (preprocess_6d.py와 동일 로직)
    save_name = name.split('/')[-1] if '/' in name else name
    npy_path = os.path.join(data6d_dir, split, f'{save_name}.npy')
    
    if not os.path.exists(npy_path):
        return None, None, None, None
    
    motion_240 = np.load(npy_path)  # [T, 240]
    return motion_240.astype(np.float32), text, name, None


# ============================================================
# 6D rotation npy 직접 로딩 (pkl 파싱 불필요)
# v2 디렉토리 구조:
#   How2Sign:  data6d/How2Sign/{split}/poses/{name}.npy
#   CSL-Daily: data6d/CSL-Daily/poses/{name}.npy
#   Phoenix:   data6d/Phoenix_2014T/{split}/{name}.npy (val->dev)
# ============================================================

def load_h2s_sample_6d(ann, data_dir, **kwargs):
    """How2Sign 6D npy 로드 [T, 240]"""
    name = ann['name']
    split = ann.get('split', 'train')
    npy_path = os.path.join(data_dir, split, 'poses', f'{name}.npy')
    if not os.path.exists(npy_path):
        return None, None, None, None
    motion = np.load(npy_path)
    return motion.astype(np.float32), ann.get('text', ''), name, None


def load_csl_sample_6d(ann, data_dir, **kwargs):
    """CSL-Daily 6D npy 로드 [T, 240]"""
    name = ann['name']
    npy_path = os.path.join(data_dir, 'poses', f'{name}.npy')
    if not os.path.exists(npy_path):
        return None, None, None, None
    motion = np.load(npy_path)
    return motion.astype(np.float32), ann.get('text', ''), name, None


def load_phoenix_sample_6d(ann, data_dir, **kwargs):
    """Phoenix-2014T 6D npy 로드 [T, 240]"""
    name = ann['name']
    split = ann.get('split', 'train')
    if split == 'val':
        split = 'dev'
    npy_path = os.path.join(data_dir, split, f'{name}.npy')
    if not os.path.exists(npy_path):
        npy_path = os.path.join(data_dir, f'{name}.npy')
    if not os.path.exists(npy_path):
        return None, None, None, None
    motion = np.load(npy_path)
    return motion.astype(np.float32), ann.get('text', ''), name, None


# ============================================================
# 범용 npy 로딩 (528D 등 임의 차원)
# 디렉토리 구조:
#   How2Sign:  {data_dir}/{split}/poses/{name}.npy
#   CSL-Daily: {data_dir}/poses/{name}.npy
#   Phoenix:   {data_dir}/{split}/{name}.npy
# ============================================================

def load_npy_sample(ann, data_dir, dataset_type='how2sign'):
    """범용 npy 로드 — dataset_type에 따라 경로 구조 결정"""
    name = ann['name']
    text = ann.get('text', '')
    split = ann.get('split', 'train')
    save_name = name.split('/')[-1] if '/' in name else name

    if dataset_type == 'how2sign':
        npy_path = os.path.join(data_dir, split, 'poses', f'{save_name}.npy')
    elif dataset_type == 'csl':
        npy_path = os.path.join(data_dir, 'poses', f'{save_name}.npy')
    elif dataset_type == 'phoenix':
        npy_path = os.path.join(data_dir, split, f'{save_name}.npy')
    else:
        npy_path = os.path.join(data_dir, f'{save_name}.npy')

    if not os.path.exists(npy_path):
        return None, None, None, None

    motion = np.load(npy_path)
    return motion.astype(np.float32), text, name, None

def load_npy_sample_360(ann, data_root_360, dataset_type='how2sign'):
    """360D npy 로드 — data360/{How2Sign,CSL-Daily,Phoenix_2014T}/{split}/*.npy"""
    name  = ann['name']
    text  = ann.get('text', '')
    split = ann.get('split', 'train')
    save_name = name.split('/')[-1] if '/' in name else name

    if dataset_type == 'how2sign':
        npy_path = os.path.join(data_root_360, split, f'{save_name}.npy')
    elif dataset_type == 'csl':
        npy_path = os.path.join(data_root_360, split, f'{save_name}.npy')
    elif dataset_type == 'phoenix':
        npy_path = os.path.join(data_root_360, split, f'{save_name}.npy')
    else:
        npy_path = os.path.join(data_root_360, f'{save_name}.npy')

    if not os.path.exists(npy_path):
        return None, None, None, None

    motion = np.load(npy_path)
    return motion.astype(np.float32), text, name, None


POS120_IDX = list(range(0,30)) + list(range(90,135)) + list(range(225,270))

# pos107: dead body dims 제거 (std < 0.01인 pelvis/spine/collar 등 13개 제거)
# alive body in pos120 local: [7,10,11,13,16,18-29] = 17 dims
# hands: 90 dims (unchanged)
_ALIVE_BODY_LOCAL = [7, 10, 11, 13, 16] + list(range(18, 30))
ALIVE_POS_IDX = [POS120_IDX[i] for i in _ALIVE_BODY_LOCAL + list(range(30, 120))]  # 107

def load_npy_sample_pos120(ann, data_root_360, dataset_type='how2sign'):
    """360D npy에서 position만 추출 → 107D [alive_body(17)+lhand(45)+rhand(45)]

    Dead body dims (pelvis, spine 등 near-constant) 제거됨.
    """
    motion, text, name, _ = load_npy_sample_360(ann, data_root_360, dataset_type)
    if motion is None:
        return None, None, None, None
    if motion.shape[1] >= 270:  # 360D npy
        return motion[:, ALIVE_POS_IDX].astype(np.float32), text, name, None
    else:
        return motion[:, :107].astype(np.float32), text, name, None