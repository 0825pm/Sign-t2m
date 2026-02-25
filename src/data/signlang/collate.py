"""
Collate function for Sign Language Datasets
Compatible with Light-T2M training pipeline
"""
import torch
import numpy as np
from typing import List, Dict


def sign_collate(batch: List[Dict]) -> Dict:
    """
    Collate function compatible with Light-T2M
    
    Input batch item (dict):
        - motion: [T, 120]
        - length: int
        - text: str
        - name: str
        - src: str
    
    Output (dict):
        - motion: [B, max_T, 120]
        - length: [B]
        - text: List[str]
        - name: List[str]
    """
    # Filter None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    motions = [b['motion'] for b in batch]
    lengths = [b['motion_len'] for b in batch]
    texts = [b['text'] for b in batch]
    names = [b['name'] for b in batch]
    srcs = [b.get('src', '') for b in batch]
    
    batch_size = len(batch)
    max_len = max(lengths)
    feat_dim = motions[0].shape[-1]
    
    # Pad motions
    motion_padded = torch.zeros(batch_size, max_len, feat_dim)
    for i, (motion, length) in enumerate(zip(motions, lengths)):
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion).float()
        motion_padded[i, :length] = motion[:length]
    
    return {
        'motion': motion_padded,
        'motion_len': torch.tensor(lengths),
        'text': texts,
        'name': names,
        'src': srcs,
    }
