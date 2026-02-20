"""
dry_run.py — Sign-t2m 학습 파이프라인 검증 (서버에서 실행)

데이터 없이 mock으로 전체 forward/backward pass 검증
실패 시 정확한 에러 위치 출력

Usage:
    cd ~/Projects/research/Sign-t2m
    python dry_run.py                    # GPU
    python dry_run.py --device cpu       # CPU
    python dry_run.py --skip_data        # 데이터 로딩 스킵 (모델만 테스트)
"""

import os
import sys
import time
import argparse
import traceback

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_mark(ok):
    return "✅" if ok else "❌"


def test_imports():
    """Step 1: 핵심 import 검증"""
    print("\n[1/6] Import 검증")
    results = {}

    modules = [
        ("torch", "torch"),
        ("lightning", "lightning.pytorch"),
        ("hydra", "hydra"),
        ("omegaconf", "omegaconf"),
        ("diffusers", "diffusers"),
        ("einops", "einops"),
        ("clip", "clip"),
        ("mamba_ssm", "mamba_ssm"),
        ("transformers", "transformers"),
    ]

    all_ok = True
    for name, mod in modules:
        try:
            __import__(mod)
            results[name] = True
            print(f"  {check_mark(True)} {name}")
        except ImportError as e:
            results[name] = False
            all_ok = False
            print(f"  {check_mark(False)} {name}: {e}")

    # Project modules
    proj_modules = [
        ("SignMotionGeneration", "src.models.sign_t2m", "SignMotionGeneration"),
        ("SignDenoiser", "src.models.nets.sign_denoiser", "SignDenoiser"),
        ("CLIP encoder", "src.models.nets.text_encoder", "CLIP"),
        ("embedding", "src.models.utils.embedding", "timestep_embedding"),
        ("lengths_to_mask", "src.models.utils.utils", "lengths_to_mask"),
        ("EMAModel", "src.models.nets.ema", "EMAModel"),
        ("CosineWarmupScheduler", "src.models.utils.utils", "CosineWarmupScheduler"),
    ]

    for name, mod, attr in proj_modules:
        try:
            m = __import__(mod, fromlist=[attr])
            getattr(m, attr)
            results[name] = True
            print(f"  {check_mark(True)} {name}")
        except Exception as e:
            results[name] = False
            all_ok = False
            print(f"  {check_mark(False)} {name}: {e}")

    return all_ok, results


def test_text_encoder(device):
    """Step 2: CLIP Text Encoder"""
    print(f"\n[2/6] CLIP Text Encoder (device={device})")
    try:
        from src.models.nets.text_encoder import CLIP
        enc = CLIP(freeze_lm=True).to(device)

        texts = ["a person is signing hello", "waving hand", ""]
        with torch.no_grad():
            out = enc(texts, device)

        print(f"  {check_mark(True)} text_emb shape: {out['text_emb'].shape}")  # [3, 512]
        print(f"  {check_mark(True)} hidden shape:   {out['hidden'].shape}")
        print(f"  {check_mark(True)} mask shape:     {out['mask'].shape}")

        assert out['text_emb'].shape == (3, 512), f"Expected [3, 512], got {out['text_emb'].shape}"
        return True, enc
    except Exception as e:
        print(f"  {check_mark(False)} {e}")
        traceback.print_exc()
        return False, None


def test_denoiser(device):
    """Step 3: SignDenoiser forward"""
    print(f"\n[3/6] SignDenoiser forward pass")
    try:
        from src.models.nets.sign_denoiser import SignDenoiser

        denoiser = SignDenoiser(
            motion_dim=120,
            max_motion_len=301,
            text_dim=512,
            pos_emb="cos",
            stage_dim="256*4",
            num_groups=16,
            patch_size=8,
            ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
            rms_norm=False,
            fused_add_norm=True,
        ).to(device)

        num_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        print(f"  Params: {num_params / 1e6:.2f}M")

        B, T = 4, 64
        motion = torch.randn(B, T, 120, device=device)
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        mask[0, 50:] = False  # test padding
        timestep = torch.randint(0, 1000, (B,), device=device)
        text_emb = {"text_emb": torch.randn(B, 512, device=device)}

        with torch.no_grad():
            out = denoiser(motion, mask, timestep, text_emb)

        print(f"  {check_mark(True)} input:  {motion.shape}")
        print(f"  {check_mark(True)} output: {out.shape}")
        assert out.shape == motion.shape, f"Shape mismatch: {out.shape} vs {motion.shape}"

        # Gradient test
        motion.requires_grad_(True)
        out2 = denoiser(motion, mask, timestep, text_emb)
        loss = out2.mean()
        loss.backward()
        print(f"  {check_mark(True)} backward pass OK (grad norm: {motion.grad.norm():.6f})")

        return True, denoiser
    except Exception as e:
        print(f"  {check_mark(False)} {e}")
        traceback.print_exc()
        return False, None


def test_diffusion_pipeline(device):
    """Step 4: Full diffusion training step (mock)"""
    print(f"\n[4/6] Full training step (mock data)")
    try:
        from diffusers import DDPMScheduler, UniPCMultistepScheduler
        from src.models.nets.text_encoder import CLIP
        from src.models.nets.sign_denoiser import SignDenoiser
        from src.models.sign_t2m import SignMotionGeneration

        # Build components manually (like Hydra would)
        text_encoder = CLIP(freeze_lm=True)
        denoiser = SignDenoiser(
            motion_dim=120, max_motion_len=301, text_dim=512,
            stage_dim="256*4", num_groups=16, patch_size=8,
            ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
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

        # Optimizer / LR scheduler (partial instantiation like Hydra)
        from functools import partial
        optimizer_fn = partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.0)

        from src.models.utils.utils import CosineWarmupScheduler
        lr_scheduler_fn = partial(CosineWarmupScheduler, T_max=10, warmup=0, eta_min=1e-5)

        model = SignMotionGeneration(
            text_encoder=text_encoder,
            denoiser=denoiser,
            noise_scheduler=noise_scheduler,
            sample_scheduler=sample_scheduler,
            text_replace_prob=0.2,
            guidance_scale=4.0,
            dataset_name="how2sign_csl_phoenix",
            optimizer=optimizer_fn,
            lr_scheduler=lr_scheduler_fn,
            ema={"use_ema": False, "ema_decay": 0.99, "ema_start": 1000},
            step_num=10,
        ).to(device)

        print(f"  {check_mark(True)} SignMotionGeneration created")

        # Mock batch
        B, T = 4, 64
        batch = {
            "motion": torch.randn(B, T, 120, device=device),
            "motion_len": torch.tensor([64, 48, 60, 32], device=device),
            "text": ["signing hello", "waving hand", "pointing up", "thumbs up"],
            "name": ["s1", "s2", "s3", "s4"],
        }

        # Training step
        model.train()
        loss = model.training_step(batch, batch_idx=0)
        print(f"  {check_mark(True)} training_step loss: {loss.item():.6f}")

        # Backward
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.denoiser.parameters() if p.grad is not None)
        print(f"  {check_mark(True)} backward OK (total grad norm: {grad_norm:.4f})")

        # Optimizer step
        opt = optimizer_fn(model.denoiser.parameters())
        opt.step()
        print(f"  {check_mark(True)} optimizer step OK")

        # Validation step
        model.eval()
        model.validation_step(batch, batch_idx=0)
        print(f"  {check_mark(True)} validation_step OK")

        return True, model
    except Exception as e:
        print(f"  {check_mark(False)} {e}")
        traceback.print_exc()
        return False, None


def test_generation(model, device):
    """Step 5: Inference (sample_motion)"""
    print(f"\n[5/6] Generation (sample_motion)")
    try:
        model.eval()
        B, T = 2, 48
        gt_motion = torch.randn(B, T, 120, device=device)
        lengths = torch.tensor([48, 32], device=device)
        texts = ["hello world", "waving"]

        t0 = time.time()
        with torch.no_grad():
            generated = model.sample_motion(gt_motion, lengths, texts)
        dt = time.time() - t0

        print(f"  {check_mark(True)} output shape: {generated.shape}")
        print(f"  {check_mark(True)} time: {dt:.2f}s")
        print(f"  {check_mark(True)} output range: [{generated.min():.4f}, {generated.max():.4f}]")

        # Check padding is zeroed
        assert (generated[1, 32:] == 0).all(), "Padding not zeroed!"
        print(f"  {check_mark(True)} padding zeroed correctly")

        return True
    except Exception as e:
        print(f"  {check_mark(False)} {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Step 6: DataModule 로딩 (실제 데이터 필요)"""
    print(f"\n[6/6] DataModule 로딩")
    try:
        from src.data.sign_datamodule import SignDataModule

        dm = SignDataModule(
            data_root="/home/user/Projects/research/SOKE/data/How2Sign",
            csl_root="/home/user/Projects/research/SOKE/data/CSL-Daily",
            phoenix_root="/home/user/Projects/research/SOKE/data/Phoenix_2014T",
            mean_path="/home/user/Projects/research/SOKE/data/CSL-Daily/mean_120.pt",
            std_path="/home/user/Projects/research/SOKE/data/CSL-Daily/std_120.pt",
            csl_mean_path="/home/user/Projects/research/SOKE/data/CSL-Daily/csl_mean_120.pt",
            csl_std_path="/home/user/Projects/research/SOKE/data/CSL-Daily/csl_std_120.pt",
            batch_size=8,
            num_workers=0,
            nfeats=120,
            dataset_name="how2sign_csl_phoenix",
            stage="t2m",
            max_motion_length=300,
            min_motion_length=40,
        )

        print(f"  {check_mark(True)} DataModule created")

        dm.setup("fit")
        print(f"  {check_mark(True)} setup('fit') OK")
        print(f"    train: {len(dm.train_dataset)}, val: {len(dm.val_dataset)}")

        # --- Per-source breakdown ---
        train_ds = dm.train_dataset
        src_counts = {}
        for item in train_ds.all_data:
            s = item.get('src', 'unknown')
            src_counts[s] = src_counts.get(s, 0) + 1

        expected_sources = {'how2sign', 'csl', 'phoenix'}
        actual_sources = set(src_counts.keys())
        missing = expected_sources - actual_sources

        print(f"  {check_mark(True)} Source breakdown (train):")
        for s, c in sorted(src_counts.items()):
            pct = c / len(train_ds) * 100
            print(f"    {s:12s}: {c:6d} ({pct:.1f}%)")

        if missing:
            print(f"  {check_mark(False)} Missing sources: {missing}")
        else:
            print(f"  {check_mark(True)} All 3 sources present (H2S + CSL + Phoenix)")

        # --- Sample from each source ---
        print(f"\n  Per-source sample test:")
        for src_key in sorted(actual_sources):
            src_idx = next(i for i, item in enumerate(train_ds.all_data) if item.get('src') == src_key)
            item = train_ds[src_idx]
            if item is not None:
                m = item['motion']
                print(f"    {src_key:12s}: shape={list(m.shape)}, "
                      f"len={item['motion_len']}, "
                      f"range=[{m.min():.3f}, {m.max():.3f}], "
                      f"text=\"{item['text'][:40]}...\"")
            else:
                print(f"    {check_mark(False)} {src_key}: load returned None")

        # --- Batch test ---
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        print(f"\n  {check_mark(True)} batch loaded")
        print(f"    motion:     {batch['motion'].shape}")
        print(f"    motion_len: {batch['motion_len'].tolist()}")
        print(f"    text[0]:    {batch['text'][0][:60]}...")

        # Sanity checks
        motion = batch['motion']
        assert motion.shape[-1] == 120, f"Expected 120D, got {motion.shape[-1]}"
        assert not torch.isnan(motion).any(), "NaN in motion!"
        assert not torch.isinf(motion).any(), "Inf in motion!"
        print(f"  {check_mark(True)} no NaN/Inf")
        print(f"    motion range: [{motion.min():.4f}, {motion.max():.4f}]")
        print(f"    motion std:   {motion.std():.4f}")

        return True
    except FileNotFoundError as e:
        print(f"  ⚠️  데이터 경로 없음 (서버에서 실행 필요): {e}")
        return None  # not a failure, just missing data
    except Exception as e:
        print(f"  {check_mark(False)} {e}")
        traceback.print_exc()
        return False


def test_hydra_config():
    """Bonus: Hydra config resolution 테스트"""
    print(f"\n[Bonus] Hydra config resolution")
    try:
        from omegaconf import OmegaConf
        from hydra import compose, initialize_config_dir

        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="train", overrides=["trainer.max_epochs=5"])

        print(f"  {check_mark(True)} Config resolved")
        print(f"    model._target_:   {cfg.model._target_}")
        print(f"    data._target_:    {cfg.data._target_}")
        print(f"    denoiser._target_: {cfg.model.denoiser._target_}")
        print(f"    optimizer.lr:     {cfg.model.optimizer.lr}")
        print(f"    lr_scheduler:     {cfg.model.lr_scheduler._target_}")
        print(f"    noise prediction: {cfg.model.noise_scheduler.prediction_type}")
        print(f"    max_epochs:       {cfg.trainer.max_epochs}")
        return True
    except Exception as e:
        print(f"  {check_mark(False)} {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Sign-t2m Dry Run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_data", action="store_true", help="데이터 로딩 스킵")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Sign-t2m Dry Run — device: {device}")
    print("=" * 60)

    results = {}

    # 1. Imports
    ok, _ = test_imports()
    results["imports"] = ok
    if not ok:
        print("\n⚠️  Import 실패. 나머지 테스트 스킵.")
        return

    # 2. Text Encoder
    ok, enc = test_text_encoder(device)
    results["text_encoder"] = ok

    # 3. Denoiser
    ok, dn = test_denoiser(device)
    results["denoiser"] = ok

    # 4. Full Pipeline
    ok, model = test_diffusion_pipeline(device)
    results["pipeline"] = ok

    # 5. Generation
    if model is not None:
        ok = test_generation(model, device)
        results["generation"] = ok

    # 6. Data Loading
    if not args.skip_data:
        ok = test_data_loading()
        results["data"] = ok
    else:
        print(f"\n[6/6] DataModule 로딩 — SKIPPED (--skip_data)")
        results["data"] = None

    # Bonus: Hydra
    ok = test_hydra_config()
    results["hydra"] = ok

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        if ok is None:
            status = "⏭️  SKIPPED"
        elif ok:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {status}  {name}")

    all_pass = all(v is True for v in results.values() if v is not None)
    print(f"\n{'🎉 All tests passed!' if all_pass else '⚠️  Some tests failed.'}")

    if all_pass:
        print(f"\n학습 시작 명령어:")
        print(f"  python src/train.py")
        print(f"  python src/train.py trainer.max_epochs=100")
        print(f"  python src/train.py trainer.devices=[0,1]  # multi-GPU")

    print("=" * 60)


if __name__ == "__main__":
    main()