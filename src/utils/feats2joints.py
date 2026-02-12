"""
Sign-t2m feats2joints - SMPLX 기반
Uses human_models.get_coord for proper SMPLX forward pass

120-dim SOKE Feature Structure:
  [0:30]    upper_body (10 joints × 3)
  [30:75]   lhand (15 joints × 3)  
  [75:120]  rhand (15 joints × 3)
"""

import torch
import torch.nn as nn

# Import SMPLX utilities from human_models
try:
    from src.utils.human_models import get_coord, smpl_x
    HAS_SMPLX = True
    print("[feats2joints] SMPL-X loaded via human_models")
except ImportError as e:
    HAS_SMPLX = False
    print(f"[feats2joints] Warning: human_models not available: {e}")

# Import 6D→axis-angle conversion
try:
    from src.utils.rotation_utils import rot_6d_to_axis_angle
    HAS_ROT_UTILS = True
except ImportError:
    HAS_ROT_UTILS = False
    print("[feats2joints] Warning: rotation_utils not available, 240-dim may fall back to feats2joints_6d")


# Default shape parameter (from SOKE)
DEFAULT_SHAPE = torch.tensor([
    -0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
    0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842
])


def feats2joints_smplx(features, mean, std):
    """
    Convert 120/240-dim SOKE features to 3D joints using SMPL-X.
    
    Args:
        features: [B, T, 120] or [B, T, 240] or [T, D] normalized features
        mean: [D] mean for denormalization
        std: [D] std for denormalization
        
    Returns:
        vertices: [B, T, 10475, 3] or None
        joints: [B, T, 144, 3] SMPLX joints
    """
    # Handle 2D input
    squeeze_output = False
    if len(features.shape) == 2:
        features = features.unsqueeze(0)
        squeeze_output = True
    
    B, T, D = features.shape
    device = features.device
    dtype = features.dtype
    
    # Denormalize
    mean = mean.to(device).to(dtype)
    std = std.to(device).to(dtype)
    features = features * std + mean
    
    if not HAS_SMPLX:
        # Fallback to approximate (zeros for missing joints)
        print("[feats2joints] Warning: SMPLX not available, using approximate")
        joints = _approximate_joints(features)
        if squeeze_output:
            joints = joints.squeeze(0)
        return None, joints
    
    # Reshape for batch processing
    features_flat = features.reshape(B * T, D)
    batch_size = B * T
    
    # ---- 240-dim 6D → 120-dim axis-angle 변환 후 기존 로직 재사용 ----
    if D == 240:
        if not HAS_ROT_UTILS:
            print("[feats2joints] Error: rotation_utils required for 240-dim")
            joints = _approximate_joints(features)
            if squeeze_output:
                joints = joints.squeeze(0)
            return None, joints
        
        motion_6d = features_flat.reshape(batch_size, 40, 6)
        motion_aa = rot_6d_to_axis_angle(motion_6d)         # [B*T, 40, 3]
        features_flat = motion_aa.reshape(batch_size, 120)
        D = 120  # 이후 120-dim 로직으로 진행
    
    # Parse 120-dim features
    if D == 120:
        upper_body_pose = features_flat[:, 0:30]    # 10 joints × 3
        lhand_pose = features_flat[:, 30:75]         # 15 joints × 3
        rhand_pose = features_flat[:, 75:120]        # 15 joints × 3
        
        # Zero padding for lower body (11 joints × 3 = 33)
        lower_body_zeros = torch.zeros(batch_size, 33, device=device, dtype=dtype)
        body_pose = torch.cat([lower_body_zeros, upper_body_pose], dim=-1)  # [B*T, 63]
        
        # Zero for other parts
        root_pose = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        jaw_pose = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        expr = torch.zeros(batch_size, 10, device=device, dtype=dtype)
        
    elif D == 133:
        # 133-dim: upper_body(30) + lhand(45) + rhand(45) + jaw(3) + expr(10)
        upper_body_pose = features_flat[:, 0:30]
        lhand_pose = features_flat[:, 30:75]
        rhand_pose = features_flat[:, 75:120]
        jaw_pose = features_flat[:, 120:123]
        expr = features_flat[:, 123:133]
        
        lower_body_zeros = torch.zeros(batch_size, 33, device=device, dtype=dtype)
        body_pose = torch.cat([lower_body_zeros, upper_body_pose], dim=-1)
        root_pose = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    else:
        print(f"[feats2joints] Warning: Unsupported dim {D}, using approximate")
        joints = _approximate_joints(features)
        if squeeze_output:
            joints = joints.squeeze(0)
        return None, joints
    
    # Shape parameters
    shape_param = DEFAULT_SHAPE.to(device).to(dtype)
    shape_param = shape_param.unsqueeze(0).expand(batch_size, -1)
    
    # Forward SMPLX via get_coord
    try:
        # get_coord uses .cuda() internally, so ensure inputs are on cuda
        cuda_device = torch.device('cuda:0')
        vertices, joints = get_coord(
            root_pose=root_pose.to(cuda_device),
            body_pose=body_pose.to(cuda_device),
            lhand_pose=lhand_pose.to(cuda_device),
            rhand_pose=rhand_pose.to(cuda_device),
            jaw_pose=jaw_pose.to(cuda_device),
            shape=shape_param.to(cuda_device),
            expr=expr.to(cuda_device)
        )
        
        # Move back to original device
        joints = joints.to(device)
        if vertices is not None:
            vertices = vertices.to(device)
        
        # Reshape back to [B, T, ...]
        vertices = vertices.reshape(B, T, -1, 3) if vertices is not None else None
        joints = joints.reshape(B, T, -1, 3)
        
    except Exception as e:
        print(f"[feats2joints] SMPLX forward failed: {e}")
        joints = _approximate_joints(features)
        vertices = None
    
    # Squeeze if input was 2D
    if squeeze_output:
        if vertices is not None:
            vertices = vertices.squeeze(0)
        joints = joints.squeeze(0)
    
    return vertices, joints


def _approximate_joints(features):
    """Fallback: approximate joint positions from features (no SMPLX)"""
    B, T, D = features.shape
    device = features.device
    dtype = features.dtype
    
    if D >= 120:
        upper_body = features[..., 0:30].reshape(B, T, 10, 3)
        lhand = features[..., 30:75].reshape(B, T, 15, 3)
        rhand = features[..., 75:120].reshape(B, T, 15, 3)
        
        # Create 55-joint skeleton
        joints = torch.zeros(B, T, 55, 3, device=device, dtype=dtype)
        joints[:, :, 12:22, :] = upper_body  # upper body
        joints[:, :, 25:40, :] = lhand       # left hand
        joints[:, :, 40:55, :] = rhand       # right hand
    else:
        joints = torch.zeros(B, T, 55, 3, device=device, dtype=dtype)
    
    return joints


class Feats2Joints(nn.Module):
    """Module wrapper for feats2joints_smplx"""
    
    def __init__(self, mean=None, std=None):
        super().__init__()
        if mean is not None:
            self.register_buffer('mean', mean)
        else:
            self.register_buffer('mean', torch.zeros(120))
        if std is not None:
            self.register_buffer('std', std)
        else:
            self.register_buffer('std', torch.ones(120))
    
    def forward(self, features):
        _, joints = feats2joints_smplx(features, self.mean, self.std)
        return joints

    
def feats2joints_6d(features_6d, mean_6d=None, std_6d=None):
    """
    Convert 240-dim 6D rotation features to 3D joints.

    Flow: 6D (240) → denormalize → axis-angle (120) → feats2joints_smplx

    Args:
        features_6d: [B, T, 240] or [T, 240]  (normalized or raw)
        mean_6d: [240] mean for denormalization (None = raw input)
        std_6d:  [240] std for denormalization  (None = raw input)

    Returns:
        vertices, joints  (same as feats2joints_smplx)
    """
    from src.utils.rotation_utils import convert_240_to_120

    squeeze = False
    if len(features_6d.shape) == 2:
        features_6d = features_6d.unsqueeze(0)
        squeeze = True

    device = features_6d.device
    dtype = features_6d.dtype

    # 1. Denormalize if mean/std provided
    if mean_6d is not None and std_6d is not None:
        mean_6d = mean_6d.to(device).to(dtype)
        std_6d = std_6d.to(device).to(dtype)
        features_6d = features_6d * std_6d + mean_6d

    # 2. 6D → axis-angle
    features_aa = convert_240_to_120(features_6d)  # [B, T, 120]

    # 3. feats2joints_smplx with identity normalization (already raw)
    zero_mean = torch.zeros(120, device=device, dtype=dtype)
    one_std = torch.ones(120, device=device, dtype=dtype)
    vertices, joints = feats2joints_smplx(features_aa, zero_mean, one_std)

    if squeeze:
        if vertices is not None:
            vertices = vertices.squeeze(0)
        joints = joints.squeeze(0)

    return vertices, joints