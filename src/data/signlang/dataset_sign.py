"""
SOKE-style Sign Language Dataset
"""
import os
import gzip
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm

from .load_data import (
    load_h2s_sample, load_csl_sample, load_phoenix_sample,
    load_h2s_sample_6d, load_csl_sample_6d, load_phoenix_sample_6d,
    load_h2s_sample_133, load_csl_sample_133, load_phoenix_sample_133,
    load_npy_sample_360,
    load_npy_sample_pos120,
    load_npy_sample,
)


# Bad How2Sign samples to skip
bad_how2sign_ids = [
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
    'g3Cc_1-V31U_12-3-rgb_front'
]


class SignMotionDataset(Dataset):
    """Sign Motion Dataset for VAE training"""
    
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        nfeats=120,
        dataset_name='how2sign_csl_phoenix',
        max_motion_length=300,
        min_motion_length=40,
        unit_length=4,
        fps=25,
        csl_root=None,
        phoenix_root=None,
        npy_root=None,
        csl_npy_root=None,
        phoenix_npy_root=None,
        csl_mean=None,
        csl_std=None,
        phoenix_mean=None,
        phoenix_std=None,
        **kwargs
    ):
        self.dataset_name = dataset_name
        self.root_dir = data_root
        self.csl_root = csl_root
        self.phoenix_root = phoenix_root
        self.npy_root = npy_root                  # 528d npy 경로 (None이면 data_root 사용)
        self.csl_npy_root = csl_npy_root          # 528d CSL npy
        self.phoenix_npy_root = phoenix_npy_root  # 528d Phoenix npy
        self.split = split
        
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.nfeats = nfeats
        
        # H2S/Phoenix용 mean/std (기본)
        self.mean = mean[:self.nfeats] if len(mean) > self.nfeats else mean
        self.std = std[:self.nfeats] if len(std) > self.nfeats else std
        
        # CSL 전용 mean/std (없으면 기본값 사용)
        if csl_mean is not None:
            self.csl_mean = csl_mean[:self.nfeats] if len(csl_mean) > self.nfeats else csl_mean
        else:
            self.csl_mean = self.mean
        if csl_std is not None:
            self.csl_std = csl_std[:self.nfeats] if len(csl_std) > self.nfeats else csl_std
        else:
            self.csl_std = self.std
        if phoenix_mean is not None:
            self.phoenix_mean = phoenix_mean[:self.nfeats] if len(phoenix_mean) > self.nfeats else phoenix_mean
        else:
            self.phoenix_mean = self.mean
        if phoenix_std is not None:
            self.phoenix_std = phoenix_std[:self.nfeats] if len(phoenix_std) > self.nfeats else phoenix_std
        else:
            self.phoenix_std = self.std
        
        # numpy 변환 (캐싱)
        self.mean_np = self.mean.numpy() if isinstance(self.mean, torch.Tensor) else self.mean
        self.std_np = self.std.numpy() if isinstance(self.std, torch.Tensor) else self.std
        self.csl_mean_np = self.csl_mean.numpy() if isinstance(self.csl_mean, torch.Tensor) else self.csl_mean
        self.csl_std_np = self.csl_std.numpy() if isinstance(self.csl_std, torch.Tensor) else self.csl_std
        self.phoenix_mean_np = self.phoenix_mean.numpy() if isinstance(self.phoenix_mean, torch.Tensor) else self.phoenix_mean
        self.phoenix_std_np = self.phoenix_std.numpy() if isinstance(self.phoenix_std, torch.Tensor) else self.phoenix_std
        
        self.all_data = []
        self.h2s_len = 0
        self.csl_len = 0
        self.phoenix_len = 0
        
        self._load_annotations()
        
        mode_str = ' [6D mode]' if self.nfeats == 240 else ''
        print(f'Data loading done. All: {len(self.all_data)}, '
              f'H2S: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}{mode_str}')
    
    def _get_mean_std_np(self, src):
        """데이터셋별 mean/std 반환"""
        if src == 'csl':
            return self.csl_mean_np, self.csl_std_np
        elif src == 'phoenix':
            return self.phoenix_mean_np, self.phoenix_std_np
        else:
            return self.mean_np, self.std_np
    
    def _load_annotations(self):
        split = self.split
        
        # How2Sign
        if 'how2sign' in self.dataset_name and self.root_dir:
            csv_path = os.path.join(self.root_dir, split, 're_aligned',
                                    f'how2sign_realigned_{split}_preprocessed_fps.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(self.root_dir, split, 'preprocessed_fps.csv')
            
            if os.path.exists(csv_path):
                csv = pd.read_csv(csv_path)
                csv['DURATION'] = csv['END_REALIGNED'] - csv['START_REALIGNED']
                csv = csv[csv['DURATION'] < 30].reset_index(drop=True)
                
                print(f'{split}--loading how2sign annotations... {len(csv)}')
                for idx in tqdm(range(len(csv)), desc='How2Sign', leave=False):
                    name = csv.iloc[idx]['SENTENCE_NAME']
                    if name in bad_how2sign_ids:
                        continue
                    self.all_data.append({
                        'name': name,
                        'fps': csv.iloc[idx]['fps'],
                        'text': csv.iloc[idx]['SENTENCE'],
                        'src': 'how2sign',
                        'split': split
                    })
                self.h2s_len = len(self.all_data)
        
        # CSL-Daily
        if 'csl' in self.dataset_name and self.csl_root:
            ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
            if split == 'val':
                ann_path = os.path.join(self.csl_root, 'csl_clean.val')
            
            if os.path.exists(ann_path):
                try:
                    with gzip.open(ann_path, 'rb') as f:
                        ann = pickle.load(f)
                    print(f'{split}--loading csl annotations... {len(ann)}')
                    for item in tqdm(ann, desc='CSL-Daily', leave=False):
                        item_copy = deepcopy(item)
                        item_copy['src'] = 'csl'
                        self.all_data.append(item_copy)
                    self.csl_len = len(ann)
                except Exception as e:
                    print(f'Failed to load CSL: {e}')
        
        # Phoenix-2014T
        if 'phoenix' in self.dataset_name and self.phoenix_root:
            if split == 'val':
                ann_path = os.path.join(self.phoenix_root, 'phoenix14t.dev')
            else:
                ann_path = os.path.join(self.phoenix_root, f'phoenix14t.{split}')
            
            if os.path.exists(ann_path):
                try:
                    with gzip.open(ann_path, 'rb') as f:
                        ann = pickle.load(f)
                    print(f'{split}--loading phoenix annotations... {len(ann)}')
                    for item in tqdm(ann, desc='Phoenix', leave=False):
                        item_copy = deepcopy(item)
                        item_copy['src'] = 'phoenix'
                        self.all_data.append(item_copy)
                    self.phoenix_len = len(ann)
                except Exception as e:
                    print(f'Failed to load Phoenix: {e}')
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src = sample['src']
        name = sample['name']
        
        # 6D (nfeats=240) vs 528D/523D npy (nfeats=523,528) vs 133D pkl vs 기존 (nfeats=120) 분기
        use_6d   = (self.nfeats == 240)
        use_528d = (self.nfeats in (133, 523, 528))  # 133도 npy 로더 사용
        use_133d = (self.nfeats == 133)
        use_360d = (self.nfeats == 360)
        use_pos120 = (self.nfeats == 120 and self.npy_root is not None)
        orig_name = name

        if use_pos120:
            npy360_h2s     = self.npy_root
            npy360_csl     = self.csl_npy_root or self.csl_root
            npy360_phoenix = self.phoenix_npy_root or self.phoenix_root
            if src == 'how2sign':
                clip_poses, text, name, _ = load_npy_sample_pos120(sample, npy360_h2s, 'how2sign')
            elif src == 'csl':
                clip_poses, text, name, _ = load_npy_sample_pos120(sample, npy360_csl, 'csl')
            elif src == 'phoenix':
                clip_poses, text, name, _ = load_npy_sample_pos120(sample, npy360_phoenix, 'phoenix')
            else:
                clip_poses, text = None, ""
        elif use_360d:
            # 360D: npy_root(data360)에서 로드, annotation은 원본 root_dir에서
            npy360_h2s     = self.npy_root or self.root_dir
            npy360_csl     = self.csl_npy_root or self.csl_root
            npy360_phoenix = self.phoenix_npy_root or self.phoenix_root
            if src == 'how2sign':
                clip_poses, text, name, _ = load_npy_sample_360(sample, npy360_h2s, 'how2sign')
            elif src == 'csl':
                clip_poses, text, name, _ = load_npy_sample_360(sample, npy360_csl, 'csl')
            elif src == 'phoenix':
                clip_poses, text, name, _ = load_npy_sample_360(sample, npy360_phoenix, 'phoenix')
            else:
                clip_poses, text = None, ""
        elif use_528d:
            # 528D: npy_root에서 로드 (annotations은 기존 경로)
            if src == 'how2sign':
                clip_poses, text, name, _ = load_npy_sample(
                    sample, self.npy_root or self.root_dir, 'how2sign')
            elif src == 'csl':
                clip_poses, text, name, _ = load_npy_sample(
                    sample, self.csl_npy_root or self.csl_root, 'csl')
            elif src == 'phoenix':
                clip_poses, text, name, _ = load_npy_sample(
                    sample, self.phoenix_npy_root or self.phoenix_root, 'phoenix')
            else:
                clip_poses, text = None, ""

        elif src == 'how2sign':
            if use_6d:
                clip_poses, text, name, _ = load_h2s_sample_6d(sample, self.root_dir)
            else:
                clip_poses, text, name, _ = load_h2s_sample(sample, self.root_dir)
        elif src == 'csl':
            if use_6d:
                clip_poses, text, name, _ = load_csl_sample_6d(sample, self.csl_root)
            else:
                clip_poses, text, name, _ = load_csl_sample(sample, self.csl_root)
        elif src == 'phoenix':
            if use_6d:
                clip_poses, text, name, _ = load_phoenix_sample_6d(sample, self.phoenix_root)
            else:
                clip_poses, text, name, _ = load_phoenix_sample(sample, self.phoenix_root)
        else:
            clip_poses, text = None, ""
        
        # load 실패 시 원래 name 복원
        if name is None:
            name = orig_name
        
        if clip_poses is None:
            clip_poses = np.zeros((self.min_motion_length, self.nfeats), dtype=np.float32)
            text = ""
        
        # 데이터셋별 mean/std로 정규화 (std floor=0.01 for numerical stability)
        mean_np, std_np = self._get_mean_std_np(src)
        clip_poses = (clip_poses - mean_np) / np.maximum(std_np, 0.01)
        
        # Adjust length
        m_length = clip_poses.shape[0]
        if m_length < self.min_motion_length:
            idx_arr = np.linspace(0, m_length - 1, num=self.min_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        elif m_length > self.max_motion_length:
            idx_arr = np.linspace(0, m_length - 1, num=self.max_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            start_idx = (clip_poses.shape[0] - m_length) // 2
            clip_poses = clip_poses[start_idx:start_idx + m_length]
        
        m_length = clip_poses.shape[0]
        
        return {
            'motion': torch.from_numpy(clip_poses).float(),
            'motion_len': m_length,
            'text': text,
            'name': name,
            'src': src,
        }


class SignText2MotionDataset(SignMotionDataset):
    """Sign Text-to-Motion Dataset for T2M training"""
    
    def __init__(self, max_text_len=40, **kwargs):
        super().__init__(**kwargs)
        self.max_text_len = max_text_len


class SignText2MotionDatasetEval(SignText2MotionDataset):
    """Evaluation dataset with all_captions"""
    
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        result['all_captions'] = [result['text']] * 3
        return result