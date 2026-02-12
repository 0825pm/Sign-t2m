"""
src/utils/rotation_utils.py — 6D Rotation ↔ Axis-Angle 변환 유틸리티

6D representation (Zhou et al. CVPR 2019):
    rotation matrix의 첫 두 column을 flatten → [r00,r10,r20, r01,r11,r21]

Usage:
    from src.utils.rotation_utils import convert_240_to_120, convert_120_to_240
"""

import numpy as np
import torch


# =============================================================================
# Numpy (전처리 / 시각화용)
# =============================================================================

def rotation_6d_to_matrix_np(rot_6d):
    """
    6D → 3×3 rotation matrix (Gram-Schmidt). Numpy.

    Args:
        rot_6d: [..., 6]
    Returns:
        R: [..., 3, 3]
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)  # [..., 3, 3] columns


def matrix_to_axis_angle_np(R):
    """
    3×3 rotation matrix → axis-angle. Numpy, vectorized.

    Args:
        R: [..., 3, 3]
    Returns:
        aa: [..., 3]
    """
    shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    N = R_flat.shape[0]

    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)  # [N]

    # axis = [R21-R12, R02-R20, R10-R01] / (2*sin(angle))
    axis = np.stack([
        R_flat[:, 2, 1] - R_flat[:, 1, 2],
        R_flat[:, 0, 2] - R_flat[:, 2, 0],
        R_flat[:, 1, 0] - R_flat[:, 0, 1],
    ], axis=-1)  # [N, 3]

    sin_angle = np.sin(angle)
    safe_sin = np.where(np.abs(sin_angle) < 1e-8, 1.0, sin_angle)
    axis = axis / (2.0 * safe_sin[:, None])

    # Small angle → identity → [0,0,0]
    small = np.abs(angle) < 1e-8
    axis[small] = 0.0
    angle[small] = 0.0

    aa = axis * angle[:, None]
    return aa.reshape(shape + (3,))


def rotation_6d_to_axis_angle_np(rot_6d):
    """
    6D rotation → axis-angle. Numpy.

    Args:
        rot_6d: [..., 6]
    Returns:
        aa: [..., 3]
    """
    R = rotation_6d_to_matrix_np(rot_6d)
    return matrix_to_axis_angle_np(R)


def convert_240_to_120_np(motion_6d):
    """
    240-dim 6D → 120-dim axis-angle. Numpy.

    Args:
        motion_6d: [T, 240]
    Returns:
        motion_aa: [T, 120]
    """
    T = motion_6d.shape[0]
    joints_6d = motion_6d.reshape(T, 40, 6)
    joints_aa = rotation_6d_to_axis_angle_np(joints_6d)  # [T, 40, 3]
    return joints_aa.reshape(T, 120).astype(np.float32)


def convert_120_to_240_np(motion_aa):
    """
    120-dim axis-angle → 240-dim 6D. Numpy.
    (preprocess_6d.py의 axis_angle_to_6d와 동일)

    Args:
        motion_aa: [T, 120]
    Returns:
        motion_6d: [T, 240]
    """
    T = motion_aa.shape[0]
    joints = motion_aa.reshape(T, 40, 3)

    angle = np.linalg.norm(joints, axis=-1, keepdims=True)  # [T, 40, 1]
    safe_angle = np.where(angle < 1e-8, 1.0, angle)
    axis = joints / safe_angle

    c = np.cos(angle)
    s = np.sin(angle)
    omc = 1 - c

    kx, ky, kz = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]

    # First two columns of rotation matrix
    r00 = c + kx * kx * omc
    r10 = ky * kx * omc + kz * s
    r20 = kz * kx * omc - ky * s
    r01 = kx * ky * omc - kz * s
    r11 = c + ky * ky * omc
    r21 = kz * ky * omc + kx * s

    rot_6d = np.concatenate([r00, r10, r20, r01, r11, r21], axis=-1)

    # Small angle → identity
    small = (angle < 1e-8).squeeze(-1)
    rot_6d[small] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)

    return rot_6d.reshape(T, 240).astype(np.float32)


# =============================================================================
# Torch (학습 / 추론용)
# =============================================================================

def rotation_6d_to_matrix(rot_6d):
    """
    6D → 3×3 rotation matrix. Torch.

    Args:
        rot_6d: [..., 6]
    Returns:
        R: [..., 3, 3]
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]


def matrix_to_axis_angle(R):
    """
    3×3 rotation matrix → axis-angle. Torch.

    Args:
        R: [..., 3, 3]
    Returns:
        aa: [..., 3]
    """
    # quaternion 경유 (수치 안정성)
    return _quaternion_to_axis_angle(_matrix_to_quaternion(R))


def rotation_6d_to_axis_angle(rot_6d):
    """
    6D → axis-angle. Torch.

    Args:
        rot_6d: [..., 6]
    Returns:
        aa: [..., 3]
    """
    R = rotation_6d_to_matrix(rot_6d)
    return matrix_to_axis_angle(R)


def convert_240_to_120(motion_6d):
    """
    240-dim 6D → 120-dim axis-angle. Torch.

    Args:
        motion_6d: [B, T, 240] or [T, 240]
    Returns:
        motion_aa: same shape but last dim 120
    """
    orig_shape = motion_6d.shape[:-1]
    joints_6d = motion_6d.reshape(*orig_shape, 40, 6)
    joints_aa = rotation_6d_to_axis_angle(joints_6d)  # [..., 40, 3]
    return joints_aa.reshape(*orig_shape, 120)


def convert_120_to_240(motion_aa):
    """
    120-dim axis-angle → 240-dim 6D. Torch.

    Args:
        motion_aa: [B, T, 120] or [T, 120]
    Returns:
        motion_6d: same shape but last dim 240
    """
    orig_shape = motion_aa.shape[:-1]
    joints = motion_aa.reshape(*orig_shape, 40, 3)
    R = _axis_angle_to_matrix(joints)  # [..., 40, 3, 3]
    # 6D = first two columns: [R[:,0], R[:,1]] flattened
    rot_6d = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)  # [..., 40, 6]
    return rot_6d.reshape(*orig_shape, 240)


# =============================================================================
# Internal helpers (torch)
# =============================================================================

def _axis_angle_to_matrix(axis_angle):
    """axis-angle [..., 3] → rotation matrix [..., 3, 3]"""
    return _quaternion_to_matrix(_axis_angle_to_quaternion(axis_angle))


def _axis_angle_to_quaternion(axis_angle):
    """axis-angle [..., 3] → quaternion [..., 4] (w, x, y, z)"""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small = angles.abs() < eps
    sin_half_over_angle = torch.where(
        small, 0.5 - angles * angles / 48, torch.sin(half_angles) / angles
    )
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_over_angle], dim=-1)
    return quaternions


def _quaternion_to_matrix(quaternions):
    """quaternion [..., 4] (w,x,y,z) → rotation matrix [..., 3, 3]"""
    r, i, j, k = quaternions.unbind(-1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack([
        1 - two_s * (j*j + k*k), two_s * (i*j - k*r), two_s * (i*k + j*r),
        two_s * (i*j + k*r), 1 - two_s * (i*i + k*k), two_s * (j*k - i*r),
        two_s * (i*k - j*r), two_s * (j*k + i*r), 1 - two_s * (i*i + j*j),
    ], dim=-1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _matrix_to_quaternion(matrix):
    """rotation matrix [..., 3, 3] → quaternion [..., 4] (w,x,y,z)"""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}")

    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    def safe_sqrt(x):
        return torch.sqrt(torch.clamp(x, min=0.0))

    t = m00 + m11 + m22
    w = safe_sqrt(1 + t) / 2
    x = safe_sqrt(1 + m00 - m11 - m22) / 2
    y = safe_sqrt(1 - m00 + m11 - m22) / 2
    z = safe_sqrt(1 - m00 - m11 + m22) / 2

    x = torch.copysign(x, m21 - m12)
    y = torch.copysign(y, m02 - m20)
    z = torch.copysign(z, m10 - m01)

    return torch.stack([w, x, y, z], dim=-1)


def _quaternion_to_axis_angle(quaternions):
    """quaternion [..., 4] (w,x,y,z) → axis-angle [..., 3]"""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small = angles.abs() < eps
    sin_half_over_angle = torch.where(
        small, 0.5 - angles * angles / 48, torch.sin(half_angles) / angles
    )
    return quaternions[..., 1:] / sin_half_over_angle
