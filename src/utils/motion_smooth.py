"""
src/utils/motion_smooth.py — Velocity-based Motion Smoothing

포즈 추정 실패로 인한 비정상적 spike만 제거하고,
정상적인 빠른 수어 동작은 보존.

전략:
    1. 각 joint별 frame-to-frame angular velocity 계산
    2. threshold 초과 frame을 spike로 판정
    3. spike frame만 linear interpolation으로 교체
    4. 나머지는 절대 안 건드림

Usage:
    from src.utils.motion_smooth import smooth_motion

    motion_120, stats = smooth_motion(raw_120)  # [T, 120] axis-angle
"""

import numpy as np


# =============================================================================
# Thresholds (rad/frame @ 25fps)
#
# 25fps → 1 frame = 40ms
# body:  max ~86°/frame = 1.5 rad  (실제 상체 회전 한계)
# hand:  max ~143°/frame = 2.5 rad (손가락은 더 빠름)
#
# 포즈 추정 실패 시 보통 >180° 점프 → 확실히 잡힘
# =============================================================================

DEFAULT_BODY_THRESH = 1.5   # rad/frame
DEFAULT_HAND_THRESH = 2.5   # rad/frame

# 120-dim joint groups: (start, end, n_joints)
JOINT_GROUPS = {
    'body':  (0,   30,  10),   # upper body: 10 joints × 3
    'lhand': (30,  75,  15),   # left hand:  15 joints × 3
    'rhand': (75,  120, 15),   # right hand: 15 joints × 3
}


def smooth_motion(motion_aa, body_thresh=DEFAULT_BODY_THRESH,
                  hand_thresh=DEFAULT_HAND_THRESH, verbose=False):
    """
    Velocity-based spike removal + linear interpolation.

    수어 동작은 보존하면서 포즈 추정 실패 spike만 제거.

    Args:
        motion_aa: [T, 120] axis-angle features (raw, NOT normalized)
        body_thresh: body joints angular velocity threshold (rad/frame)
        hand_thresh: hand joints angular velocity threshold (rad/frame)
        verbose: print per-joint spike info

    Returns:
        smoothed: [T, 120] smoothed motion
        stats: dict with smoothing statistics
    """
    T, D = motion_aa.shape
    assert D == 120, f"Expected 120-dim, got {D}"

    if T < 3:
        return motion_aa.copy(), {'total_fixed': 0, 'total_frames': T}

    smoothed = motion_aa.copy()

    # ---- Step 0: Zero/dead frame detection ----
    # pkl 로딩 실패 시 해당 프레임이 전부 0 → 이웃 기준으로 interpolation
    n_zero_fixed = _fix_dead_frames(smoothed, verbose)

    total_fixed = 0
    per_group = {}

    groups = [
        ('body',  *JOINT_GROUPS['body'],  body_thresh),
        ('lhand', *JOINT_GROUPS['lhand'], hand_thresh),
        ('rhand', *JOINT_GROUPS['rhand'], hand_thresh),
    ]

    for group_name, start, end, n_joints, thresh in groups:
        group_fixed = 0

        for j in range(n_joints):
            jstart = start + j * 3
            joint_seq = smoothed[:, jstart:jstart+3]  # [T, 3]

            # Frame-to-frame angular velocity
            vel = np.linalg.norm(np.diff(joint_seq, axis=0), axis=1)  # [T-1]

            # Spike detection: frame t is spike if vel[t-1] > thresh
            spike_mask = np.zeros(T, dtype=bool)
            spike_mask[1:] = vel > thresh

            # 추가: "왕복 spike" 감지 — 한 프레임 튀었다 돌아오는 경우
            # frame t가 spike이고, t+1이 정상이면 → t는 확실한 spike
            # frame t가 spike이고, t+1도 spike이면 → 연속 spike (둘 다 interpolate)
            # 하지만 t는 spike인데 t-1→t+1 velocity는 정상이면?
            # 이건 t가 중간에 뻗은 거니까 그대로 spike 처리
            _detect_return_spikes(smoothed, spike_mask, jstart, thresh)

            if not spike_mask.any():
                continue

            # Good frame indices (anchors for interpolation)
            good_idx = np.where(~spike_mask)[0]

            if len(good_idx) < 2:
                # 거의 전체가 spike → 이 joint는 복구 불가, 건너뜀
                if verbose:
                    print(f"    [{group_name}/j{j}] {spike_mask.sum()}/{T} spikes, "
                          f"too few anchors ({len(good_idx)}), skipping")
                continue

            # Linear interpolation (per axis-angle dimension)
            spike_idx = np.where(spike_mask)[0]
            for dim in range(3):
                smoothed[spike_idx, jstart + dim] = np.interp(
                    spike_idx,
                    good_idx,
                    joint_seq[good_idx, dim]
                )

            n_fixed = len(spike_idx)
            group_fixed += n_fixed

            if verbose and n_fixed > 0:
                max_vel = vel[spike_mask[1:]].max() if spike_mask[1:].any() else 0
                print(f"    [{group_name}/j{j}] fixed {n_fixed}/{T} frames "
                      f"(max_vel={max_vel:.2f} rad/frame)")

        per_group[group_name] = group_fixed
        total_fixed += group_fixed

    stats = {
        'total_fixed': total_fixed,
        'zero_fixed': n_zero_fixed,
        'total_frames': T,
        'fix_ratio': total_fixed / (T * 40) if T > 0 else 0,
        'per_group': per_group,
    }

    return smoothed, stats


def _fix_dead_frames(motion, verbose=False):
    """
    Dead frame 감지 및 보정.

    포즈 추정 완전 실패 → 해당 프레임 전체가 0 (또는 거의 0).
    Max absolute value 기준으로 판별 (L2보다 robust).

    Args:
        motion: [T, 120] in-place 수정
        verbose: print info
    Returns:
        n_fixed: number of dead frames fixed
    """
    T = motion.shape[0]
    if T < 3:
        return 0

    # 프레임별 max |value| — 모든 dim이 0이어야 dead
    frame_max = np.max(np.abs(motion), axis=1)  # [T]

    # 전체 시퀀스의 typical scale
    median_max = np.median(frame_max)
    if median_max < 1e-6:
        return 0  # 전체가 dead면 skip

    # Dead = 모든 dim이 사실상 0 (절대 + 상대 기준)
    abs_thresh = 0.01  # 모든 joint rotation이 0.01 rad 미만 → 사실상 dead
    rel_thresh = median_max * 0.01
    dead_thresh = max(abs_thresh, rel_thresh)
    dead_mask = frame_max < dead_thresh

    # 첫/마지막 프레임은 건드리지 않음 (anchor 필요)
    dead_mask[0] = False
    dead_mask[-1] = False

    if not dead_mask.any():
        return 0

    # Good frame으로 interpolation
    good_idx = np.where(~dead_mask)[0]
    dead_idx = np.where(dead_mask)[0]

    if len(good_idx) < 2:
        return 0

    for dim in range(motion.shape[1]):
        motion[dead_idx, dim] = np.interp(dead_idx, good_idx, motion[good_idx, dim])

    if verbose:
        print(f"    Dead frames fixed: {len(dead_idx)} / {T}")

    return len(dead_idx)


def _detect_return_spikes(motion, spike_mask, jstart, thresh):
    """
    왕복 spike 추가 감지.

    패턴: 정상 → spike → 정상 (1프레임 튀었다 돌아오는 경우)
    vel[t-1→t]는 thresh 이하일 수 있지만, 이웃과의 편차가 큰 경우.

    motion: [T, D] full motion array
    spike_mask: [T] bool, in-place로 수정
    jstart: joint 시작 인덱스
    thresh: velocity threshold
    """
    T = motion.shape[0]
    if T < 3:
        return

    joint = motion[:, jstart:jstart+3]  # [T, 3]

    for t in range(1, T - 1):
        if spike_mask[t]:
            continue  # 이미 spike로 마킹됨

        # t-1 → t velocity
        v_in = np.linalg.norm(joint[t] - joint[t-1])
        # t → t+1 velocity
        v_out = np.linalg.norm(joint[t+1] - joint[t])
        # t-1 → t+1 velocity (t를 건너뛴 경우)
        v_skip = np.linalg.norm(joint[t+1] - joint[t-1])

        # 조건: t 앞뒤 velocity 모두 크고, 건너뛰면 정상
        # → t가 단독 spike
        if v_in > thresh * 0.7 and v_out > thresh * 0.7 and v_skip < thresh * 0.5:
            spike_mask[t] = True


def smooth_motion_240(motion_6d, body_thresh=DEFAULT_BODY_THRESH,
                      hand_thresh=DEFAULT_HAND_THRESH, verbose=False):
    """
    6D (240-dim) 데이터에 smoothing 적용.
    6D → AA → smooth → 6D 라운드트립.

    이미 생성된 .npy 파일을 후처리할 때 사용.

    Args:
        motion_6d: [T, 240]
    Returns:
        smoothed_6d: [T, 240]
        stats: dict
    """
    from src.utils.rotation_utils import convert_240_to_120_np, convert_120_to_240_np

    motion_aa = convert_240_to_120_np(motion_6d)
    smoothed_aa, stats = smooth_motion(motion_aa, body_thresh, hand_thresh, verbose)
    smoothed_6d = convert_120_to_240_np(smoothed_aa)
    return smoothed_6d, stats


# =============================================================================
# Batch processing utility
# =============================================================================

def smooth_and_report(motion_aa, name='', body_thresh=DEFAULT_BODY_THRESH,
                      hand_thresh=DEFAULT_HAND_THRESH):
    """
    smooth_motion wrapper with 한 줄 요약 출력.

    Returns:
        smoothed, stats
    """
    smoothed, stats = smooth_motion(motion_aa, body_thresh, hand_thresh)

    if stats['total_fixed'] > 0 and name:
        pct = stats['total_fixed'] / (stats['total_frames'] * 40) * 100
        print(f"    {name}: fixed {stats['total_fixed']} joints "
              f"({pct:.1f}%) [body={stats['per_group'].get('body',0)}, "
              f"lhand={stats['per_group'].get('lhand',0)}, "
              f"rhand={stats['per_group'].get('rhand',0)}]")

    return smoothed, stats
