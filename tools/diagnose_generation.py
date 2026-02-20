"""
diagnose_generation.py — 생성 파이프라인 진단

테스트 항목:
1. Reconstruction test: GT에 노이즈 → 디노이즈 (모델 능력 확인)
2. Guidance 반응 테스트: scale=0 vs 7.5 vs 15 비교
3. Step 수 테스트: 10 vs 50 vs 100
4. Cond vs Uncond 차이 측정 (텍스트 영향력 확인)
5. Feature 분포 비교 (GT vs Generated)

Usage:
    python diagnose_generation.py --ckpt logs/.../last.ckpt
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NFEATS = 528
N_JOINTS = 44


def load_model(ckpt_path, device, stage_dim="384*4"):
    from src.models.sign_t2m import SignMotionGeneration
    from src.models.nets.sign_denoiser import SignDenoiser
    from src.models.nets.text_encoder import CLIP
    from diffusers import DDPMScheduler, UniPCMultistepScheduler
    from functools import partial

    ckpt = torch.load(ckpt_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})

    text_encoder = CLIP(freeze_lm=True)
    denoiser = SignDenoiser(
        motion_dim=NFEATS, max_motion_len=401, text_dim=512,
        pos_emb="cos", stage_dim=stage_dim, num_groups=16, patch_size=8,
        ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
        rms_norm=False, fused_add_norm=True,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
        clip_sample=False, prediction_type="sample",
    )
    sample_scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", solver_order=2,
        prediction_type="sample",
    )

    model = SignMotionGeneration(
        text_encoder=text_encoder, denoiser=denoiser,
        noise_scheduler=noise_scheduler, sample_scheduler=sample_scheduler,
        text_replace_prob=0.0, guidance_scale=hparams.get('guidance_scale', 7.5),
        dataset_name='how2sign',
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
        lr_scheduler=None, step_num=hparams.get('step_num', 10),
        ema=hparams.get('ema', {"use_ema": False, "ema_decay": 0.999, "ema_start": 1000}),
    )

    state = ckpt['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_te = [k for k in missing if not k.startswith('text_encoder')]
    if non_te:
        print(f"  ⚠️ Missing (non-text_encoder): {non_te[:5]}")

    # EMA 확인
    has_ema = any('ema' in k for k in state.keys())
    print(f"  EMA in checkpoint: {has_ema}")
    if model.ema_denoiser is not None:
        print(f"  EMA loaded in model: True")
    else:
        print(f"  EMA loaded in model: False")

    model.eval().to(device)
    epoch = ckpt.get('epoch', '?')
    print(f"  Epoch: {epoch}, Denoiser: {sum(p.numel() for p in denoiser.parameters())/1e6:.2f}M")
    return model


def load_val_sample(args, mean, std):
    """val set에서 1개 샘플 로드"""
    from src.data.signlang.dataset_sign import SignText2MotionDataset

    dataset = SignText2MotionDataset(
        data_root=args.data_root,
        csl_root=None, phoenix_root=None,
        npy_root=args.npy_root,
        split='val', mean=mean, std=std,
        nfeats=NFEATS, dataset_name='how2sign',
        max_motion_length=400, min_motion_length=20,
    )
    item = dataset[0]
    return item['motion'], item['text'], item['motion_len'], item['name']


def test_reconstruction(model, gt_motion_norm, text, length, device, output_dir):
    """TEST 1: GT에 다양한 레벨의 노이즈 추가 후 디노이즈"""
    print("\n" + "="*60)
    print("TEST 1: Reconstruction (GT + noise → denoise)")
    print("="*60)

    gt = gt_motion_norm.unsqueeze(0).to(device)  # [1, T, 528]
    length_t = torch.tensor([length], device=device)

    with torch.no_grad():
        text_embed = model.text_encoder([text], device)

    denoiser = model.ema_denoiser.model if model.ema_denoiser is not None else model.denoiser
    scheduler = model.noise_scheduler

    for noise_level in [50, 200, 500, 999]:
        t = torch.tensor([noise_level], device=device)
        noise = torch.randn_like(gt)
        noisy = scheduler.add_noise(gt, noise, t)

        with torch.no_grad():
            pred = denoiser(noisy, torch.ones(1, gt.shape[1], dtype=torch.bool, device=device),
                           t, text_embed)

        rmse = torch.sqrt(torch.mean((pred - gt) ** 2)).item()
        # positions만 비교
        pos_rmse = torch.sqrt(torch.mean((pred[:,:,:132] - gt[:,:,:132]) ** 2)).item()
        print(f"  t={noise_level:4d}: RMSE={rmse:.4f}, pos_RMSE={pos_rmse:.4f}")

    print("  → t=50에서 pos_RMSE < 0.05면 모델 OK, 높으면 모델 문제")


def test_guidance_response(model, text, length, device, mean_np, std_np, output_dir):
    """TEST 2: guidance_scale 변화에 따른 출력 변화 측정"""
    print("\n" + "="*60)
    print("TEST 2: Guidance Scale Response")
    print("="*60)

    B, T, D = 1, length, NFEATS
    dummy = torch.zeros(B, T, D, device=device)
    length_t = torch.tensor([length], device=device)

    results = {}
    for scale in [0.0, 1.0, 4.0, 7.5, 15.0]:
        old_scale = model.hparams.guidance_scale
        model.hparams.guidance_scale = scale

        with torch.no_grad():
            gen = model.sample_motion(dummy, length_t, [text])

        gen_np = gen[0].cpu().numpy()
        gen_raw = gen_np * (std_np + 1e-10) + mean_np
        pos = gen_raw[:, :132].reshape(-1, 44, 3)

        results[scale] = {
            'norm_std': np.std(gen_np),
            'pos_std': np.std(pos),
            'pos_range': pos.max() - pos.min(),
            'motion_var': np.mean(np.std(pos, axis=0)),  # 시간축 분산
        }
        model.hparams.guidance_scale = old_scale

        print(f"  scale={scale:5.1f}: norm_std={results[scale]['norm_std']:.4f}, "
              f"pos_range={results[scale]['pos_range']:.4f}, "
              f"temporal_var={results[scale]['motion_var']:.4f}")

    # 변화량 체크
    diff_0_75 = abs(results[0.0]['pos_std'] - results[7.5]['pos_std'])
    print(f"\n  scale=0 vs 7.5 pos_std 차이: {diff_0_75:.4f}")
    if diff_0_75 < 0.01:
        print("  ⚠️ Guidance가 거의 영향 없음 → cond ≈ uncond → 텍스트 조건 무용")
    else:
        print("  ✅ Guidance 반응 있음")


def test_cond_uncond_diff(model, text, length, device):
    """TEST 3: 동일 노이즈에서 cond vs uncond denoiser 출력 차이"""
    print("\n" + "="*60)
    print("TEST 3: Cond vs Uncond Output Difference")
    print("="*60)

    B, T, D = 1, length, NFEATS
    denoiser = model.ema_denoiser.model if model.ema_denoiser is not None else model.denoiser

    x = torch.randn(B, T, D, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    with torch.no_grad():
        # Conditional
        cond_embed = model.text_encoder([text], device)
        # Unconditional
        uncond_embed = model.text_encoder([""], device)

    for t_val in [999, 500, 100, 10]:
        t = torch.tensor([t_val], device=device)
        with torch.no_grad():
            cond_out = denoiser(x, mask, t, cond_embed)
            uncond_out = denoiser(x, mask, t, uncond_embed)

        diff = (cond_out - uncond_out).abs()
        diff_mean = diff.mean().item()
        diff_max = diff.max().item()
        # positions만
        pos_diff = diff[:, :, :132].mean().item()

        out_scale = cond_out.abs().mean().item()
        ratio = diff_mean / (out_scale + 1e-8)

        print(f"  t={t_val:4d}: diff_mean={diff_mean:.6f}, diff_max={diff_max:.4f}, "
              f"output_scale={out_scale:.4f}, diff/out_ratio={ratio:.4f}")

    print(f"\n  ratio > 0.1 이면 텍스트 영향 있음")
    print(f"  ratio < 0.01 이면 텍스트 거의 무시됨")


def test_text_encoder(model, device):
    """TEST 4: 텍스트 인코더 출력 확인"""
    print("\n" + "="*60)
    print("TEST 4: Text Encoder Output Check")
    print("="*60)

    texts = [
        "a person waves hello",
        "pointing to the right",
        "",  # unconditional
        "The Los Angeles Police Department is now employing",
    ]

    with torch.no_grad():
        for t in texts:
            embed = model.text_encoder([t], device)
            emb = embed['text_emb']     # [1, 512]
            hid = embed['hidden']       # [1, L, 512]
            mask = embed['mask']        # [1, L]

            label = f'"{t[:40]}"' if t else '"" (uncond)'
            print(f"  {label}")
            print(f"    text_emb: norm={emb.norm():.4f}, mean={emb.mean():.4f}, std={emb.std():.4f}")
            print(f"    hidden:   shape={list(hid.shape)}, norm={hid.norm():.4f}")
            print(f"    mask:     {mask.sum().item()}/{mask.shape[1]} tokens active")

    # 임베딩 간 코사인 유사도
    print(f"\n  Cosine similarities:")
    with torch.no_grad():
        embeds = [model.text_encoder([t], device)['text_emb'].squeeze() for t in texts]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            cos = torch.nn.functional.cosine_similarity(embeds[i], embeds[j], dim=0)
            t1 = texts[i][:20] if texts[i] else '""'
            t2 = texts[j][:20] if texts[j] else '""'
            print(f"    {t1} vs {t2}: cos={cos:.4f}")


def test_feature_distribution(model, gt_motion_norm, text, length, device, mean_np, std_np, output_dir):
    """TEST 5: GT vs Generated 피처 분포 비교"""
    print("\n" + "="*60)
    print("TEST 5: Feature Distribution (GT vs Generated)")
    print("="*60)

    B, T, D = 1, length, NFEATS
    dummy = torch.zeros(B, T, D, device=device)
    length_t = torch.tensor([length], device=device)

    with torch.no_grad():
        gen = model.sample_motion(dummy, length_t, [text])

    gt_np = gt_motion_norm.numpy()  # [T, 528] normalized
    gen_np = gen[0].cpu().numpy()    # [T, 528] normalized

    parts = [
        ('positions', 0, 132),
        ('velocities', 132, 264),
        ('6D_rot', 264, 528),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Feature Distribution: GT (blue) vs Generated (red)')

    for idx, (name, s, e) in enumerate(parts):
        gt_part = gt_np[:length, s:e]
        gen_part = gen_np[:length, s:e]

        print(f"\n  {name} [{s}:{e}]:")
        print(f"    GT:  mean={gt_part.mean():.4f}, std={gt_part.std():.4f}, "
              f"range=[{gt_part.min():.3f}, {gt_part.max():.3f}]")
        print(f"    Gen: mean={gen_part.mean():.4f}, std={gen_part.std():.4f}, "
              f"range=[{gen_part.min():.3f}, {gen_part.max():.3f}]")

        # Histogram
        ax = axes[0, idx]
        ax.hist(gt_part.flatten(), bins=100, alpha=0.5, color='blue', label='GT', density=True)
        ax.hist(gen_part.flatten(), bins=100, alpha=0.5, color='red', label='Gen', density=True)
        ax.set_title(f'{name} (normalized)')
        ax.legend()

        # Per-dim std
        ax = axes[1, idx]
        gt_std = gt_part.std(axis=0)
        gen_std = gen_part.std(axis=0)
        ax.plot(gt_std, 'b-', alpha=0.7, label='GT')
        ax.plot(gen_std, 'r-', alpha=0.7, label='Gen')
        ax.set_title(f'{name} per-dim std')
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_distribution.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--stage_dim', default='384*4')
    parser.add_argument('--data_root', default='/home/user/Projects/research/SOKE/data/How2Sign')
    parser.add_argument('--npy_root', default='/home/user/Projects/research/SOKE/data/How2Sign_528d')
    parser.add_argument('--mean_path', default='/home/user/Projects/research/SOKE/data/How2Sign_528d/mean_528.pt')
    parser.add_argument('--std_path', default='/home/user/Projects/research/SOKE/data/How2Sign_528d/std_528.pt')
    parser.add_argument('--output', default='diagnose_output')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    print("="*60)
    print("Sign-t2m Generation Diagnostic")
    print("="*60)

    # Load
    print("\n[Loading model...]")
    model = load_model(args.ckpt, device, args.stage_dim)

    mean = torch.load(args.mean_path, map_location='cpu').float()[:NFEATS]
    std = torch.load(args.std_path, map_location='cpu').float()[:NFEATS]
    mean_np, std_np = mean.numpy(), std.numpy()

    print("\n[Loading val sample...]")
    gt_motion, text, length, name = load_val_sample(args, mean, std)
    print(f"  Sample: {name}, T={length}, text=\"{text[:60]}...\"")

    # Run tests
    test_text_encoder(model, device)
    test_cond_uncond_diff(model, text, length, device)
    test_reconstruction(model, gt_motion, text, length, device, args.output)
    test_guidance_response(model, text, length, device, mean_np, std_np, args.output)
    test_feature_distribution(model, gt_motion, text, length, device, mean_np, std_np, args.output)

    print("\n" + "="*60)
    print("진단 완료. 결과를 공유해줘.")
    print("="*60)


if __name__ == '__main__':
    main()
