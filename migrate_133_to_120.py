"""
133-dim → 120-dim 마이그레이션 스크립트
jaw(3) + expr(10) 제거

Usage:
  cd ~/Projects/research/sign-t2m
  python migrate_133_to_120.py
"""
import torch
import os
import re

SOKE_DIR = '/home/user/Projects/research/SOKE/data/CSL-Daily'
SIGN_T2M = os.path.expanduser('~/Projects/research/sign-t2m')


def generate_mean_std():
    """133-dim mean/std에서 앞 120만 잘라서 저장"""
    print("=== 1. Mean/Std 120-dim 생성 ===")
    for name in ['mean_133', 'std_133', 'csl_mean_133', 'csl_std_133']:
        src = os.path.join(SOKE_DIR, '%s.pt' % name)
        dst = os.path.join(SOKE_DIR, '%s.pt' % name.replace('133', '120'))
        if not os.path.exists(src):
            print("  SKIP (not found): %s" % src)
            continue
        t = torch.load(src)
        t120 = t[:120]
        torch.save(t120, dst)
        print("  %s (%s) -> %s (%s)" % (name, list(t.shape), name.replace('133', '120'), list(t120.shape)))
    print()


def fix_load_data():
    """load_data.py: convert_179_to_133 → convert_179_to_120"""
    print("=== 2. load_data.py 수정 ===")
    path = os.path.join(SIGN_T2M, 'src/data/signlang/load_data.py')
    with open(path, 'r') as f:
        content = f.read()

    original = content

    # convert_179_to_133 함수를 convert_179_to_120으로 교체
    # 함수 본문 전체를 단순 슬라이싱으로 교체
    content = re.sub(
        r'def convert_179_to_133\(clip_poses\):.*?return clip_poses',
        'def convert_179_to_120(clip_poses):\n'
        '    """179-dim → 120-dim (upper_body + lhand + rhand only)\n'
        '    \n'
        '    179-dim에서 직접 추출:\n'
        '    - [36:66]   upper_body (10 joints × 3 = 30)\n'
        '    - [66:111]  lhand (15 joints × 3 = 45)\n'
        '    - [111:156] rhand (15 joints × 3 = 45)\n'
        '    \n'
        '    Total: 30 + 45 + 45 = 120\n'
        '    """\n'
        '    return clip_poses[:, 36:156]',
        content,
        flags=re.DOTALL
    )

    # 혹시 남아있는 convert_179_to_133 호출도 교체
    content = content.replace('convert_179_to_133', 'convert_179_to_120')

    # fallback zeros: 133 → 120
    content = content.replace(', 133]', ', 120]')

    if content != original:
        with open(path, 'w') as f:
            f.write(content)
        print("  수정 완료")
    else:
        print("  변경 없음 (이미 120이거나 패턴 불일치)")

    # 확인
    with open(path, 'r') as f:
        text = f.read()
    print("  convert_179_to_120 count: %d" % text.count('convert_179_to_120'))
    print("  convert_179_to_133 count: %d (0이어야 함)" % text.count('convert_179_to_133'))
    print("  ', 133]' count: %d (0이어야 함)" % text.count(', 133]'))
    print("  ', 120]' count: %d" % text.count(', 120]'))
    print()


def fix_dataset_sign():
    """dataset_sign.py: nfeats 기본값 133 → 120"""
    print("=== 3. dataset_sign.py 수정 ===")
    path = os.path.join(SIGN_T2M, 'src/data/signlang/dataset_sign.py')
    with open(path, 'r') as f:
        content = f.read()

    original = content
    content = content.replace('nfeats=133', 'nfeats=120')

    if content != original:
        with open(path, 'w') as f:
            f.write(content)
        print("  nfeats=133 → nfeats=120 수정 완료")
    else:
        print("  변경 없음 (이미 120)")
    print()


def fix_config():
    """sign_h2s.yaml 복사"""
    print("=== 4. sign_h2s.yaml 복사 ===")
    src = os.path.join(os.path.dirname(__file__), 'sign_h2s.yaml')
    if not os.path.exists(src):
        # 같은 디렉토리에 없으면 outputs에서
        src = '/mnt/user-data/outputs/sign_h2s.yaml'
    if not os.path.exists(src):
        # 직접 생성
        dst = os.path.join(SIGN_T2M, 'configs/data/sign_h2s.yaml')
        config = """_target_: src.data.sign_datamodule.SignDataModule

# Data paths (절대경로)
data_root: /home/user/Projects/research/SOKE/data/How2Sign
csl_root: /home/user/Projects/research/SOKE/data/CSL-Daily
phoenix_root: /home/user/Projects/research/SOKE/data/Phoenix_2014T

# Dataset params
dataset_name: how2sign_csl_phoenix
nfeats: 120
motion_dim: 120
njoints: 55
fps: 25

# Sequence limits
max_motion_length: 300
min_motion_length: 40

# DataLoader params
batch_size: 64
num_workers: 8
pin_memory: true

# Stage
stage: t2m

# Normalization (H2S/Phoenix용)
mean_path: /home/user/Projects/research/SOKE/data/CSL-Daily/mean_120.pt
std_path: /home/user/Projects/research/SOKE/data/CSL-Daily/std_120.pt

# CSL 전용
csl_mean_path: /home/user/Projects/research/SOKE/data/CSL-Daily/csl_mean_120.pt
csl_std_path: /home/user/Projects/research/SOKE/data/CSL-Daily/csl_std_120.pt
"""
        with open(dst, 'w') as f:
            f.write(config)
        print("  직접 생성: %s" % dst)
    else:
        import shutil
        dst = os.path.join(SIGN_T2M, 'configs/data/sign_h2s.yaml')
        shutil.copy2(src, dst)
        print("  복사: %s → %s" % (src, dst))
    print()


def verify():
    """최종 확인"""
    print("=== 5. 최종 확인 ===")

    # load_data.py
    path = os.path.join(SIGN_T2M, 'src/data/signlang/load_data.py')
    with open(path, 'r') as f:
        text = f.read()
    assert 'convert_179_to_133' not in text, "ERROR: convert_179_to_133 still exists!"
    assert 'convert_179_to_120' in text, "ERROR: convert_179_to_120 not found!"
    assert ', 133]' not in text, "ERROR: 133 zeros still exists!"
    print("  load_data.py: OK ✅")

    # dataset_sign.py
    path = os.path.join(SIGN_T2M, 'src/data/signlang/dataset_sign.py')
    with open(path, 'r') as f:
        text = f.read()
    assert 'nfeats=133' not in text, "ERROR: nfeats=133 still exists!"
    assert 'nfeats=120' in text, "ERROR: nfeats=120 not found!"
    print("  dataset_sign.py: OK ✅")

    # config
    path = os.path.join(SIGN_T2M, 'configs/data/sign_h2s.yaml')
    with open(path, 'r') as f:
        text = f.read()
    assert 'nfeats: 120' in text, "ERROR: nfeats: 120 not in config!"
    assert 'motion_dim: 120' in text, "ERROR: motion_dim: 120 not in config!"
    assert 'mean_120' in text, "ERROR: mean_120 not in config!"
    print("  sign_h2s.yaml: OK ✅")

    # mean/std files
    for name in ['mean_120', 'std_120', 'csl_mean_120', 'csl_std_120']:
        fpath = os.path.join(SOKE_DIR, '%s.pt' % name)
        if os.path.exists(fpath):
            t = torch.load(fpath)
            assert t.shape[0] == 120, "ERROR: %s shape is %s!" % (name, t.shape)
            print("  %s.pt: shape=%s ✅" % (name, list(t.shape)))
        else:
            print("  %s.pt: NOT FOUND ❌" % name)

    print()
    print("=== 마이그레이션 완료! ===")
    print()
    print("학습 커맨드:")
    print("python src/train.py trainer.devices=[0] logger=wandb data=sign_h2s \\")
    print("    data.batch_size=64 trainer.max_epochs=1000 \\")
    print("    data.dataset_name=how2sign \\")
    print("    callbacks/model_checkpoint=t2m callbacks.model_checkpoint.monitor=train/loss \\")
    print("    callbacks.model_checkpoint.mode=min +model/lr_scheduler=cosine \\")
    print("    model.guidance_scale=4 trainer.precision=32 model.evaluator=null")


if __name__ == '__main__':
    generate_mean_std()
    fix_load_data()
    fix_dataset_sign()
    fix_config()
    verify()
