"""# 기존 epsilon 스크립트에서 2줄만 수정
overfit_133d_sample.py — 133D position 1-sample overfitting test (sample prediction)

FINDINGS.md 기준:
- prediction_type=sample (x_0 직접 예측)
- UniPC는 epsilon으로 해석
- B=32, lr=1e-3, 5000 steps
- PASS: loss < 0.01
"""
import torch
torch.backends.cudnn.enabled = False
import os, sys
import numpy as np
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.nets.sign_denoiser import SignDenoiser
from src.models.nets.text_encoder import CLIP
from diffusers import DDPMScheduler, UniPCMultistepScheduler

def x0_to_eps(scheduler, x0, x_t, t):
    """sample prediction → epsilon 변환 (generate 시 사용)"""
    alpha = scheduler.alphas_cumprod[t].to(x0.device).float()
    alpha = alpha.view(-1, 1, 1)
    return (x_t - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-8)

def main():
    device = torch.device('cuda:0')
    DATA = '/home/user/Projects/research/SOKE/data/How2Sign_133d'
    npy_dir = f'{DATA}/train/poses'
    npy_file = sorted(os.listdir(npy_dir))[0]
    raw = np.load(os.path.join(npy_dir, npy_file))
    mean = torch.load(f'{DATA}/mean_133.pt', map_location='cpu').numpy()
    std  = torch.load(f'{DATA}/std_133.pt',  map_location='cpu').numpy()
    norm = (raw - mean) / (std + 1e-10)
    T = (min(norm.shape[0], 100) // 4) * 4
    motion_1 = torch.from_numpy(norm[:T]).float().unsqueeze(0).to(device)
    B = 32
    motion = motion_1.repeat(B, 1, 1)   # [32, T, 133]
    print(f"Sample: {npy_file}, T={T}, batch={B}")

    # Model
    text_encoder = CLIP().to(device).eval()
    for p in text_encoder.parameters(): p.requires_grad_(False)

    denoiser = SignDenoiser(
        motion_dim=133, max_motion_len=401,
        text_dim=512, stage_dim="256*4"
    ).to(device).train()

    # sample prediction — target은 x_0
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="squaredcos_cap_v2", clip_sample=False,
        prediction_type="sample"
    )
    print(f"Params: {sum(p.numel() for p in denoiser.parameters() if p.requires_grad)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-3)

    with torch.no_grad():
        text_dict = text_encoder(["a person signing"] * B, device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    print("\n=== sample prediction, B=32, lr=1e-3, 5000 steps ===\n")
    for step in range(5000):
        optimizer.zero_grad()
        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(motion)
        x_t = scheduler.add_noise(motion, noise, t)
        pred_x0 = denoiser(x_t, mask, t, text_dict)
        loss = F.mse_loss(pred_x0, motion)   # target = x_0 (원본 모션)
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == 4999:
            print(f"  step {step:5d}  loss={loss.item():.6f}")

    f = loss.item()
    verdict = 'PASS ✅' if f < 0.01 else 'PARTIAL ⚠️' if f < 0.1 else 'FAIL ❌'
    print(f"\n{verdict} — loss={f:.6f}")

    # =========================================================================
    # Generate & visualize (PASS이면)
    # =========================================================================
    if f < 0.5:
        print("\n=== Generating (UniPC epsilon, no CFG) ===")
        from diffusers import UniPCMultistepScheduler

        sample_scheduler = UniPCMultistepScheduler(
            num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
            beta_schedule="squaredcos_cap_v2", solver_order=2,
            prediction_type='epsilon'   # FINDINGS.md: 항상 epsilon
        )
        sample_scheduler.set_timesteps(10)

        denoiser.eval()
        with torch.no_grad():
            text_dict_1 = text_encoder(["a person signing"], device)
            # text_dict를 batch=1로
            text_1 = {k: v[:1] if isinstance(v, torch.Tensor) else v
                      for k, v in text_dict_1.items()}
            mask_1 = torch.ones(1, T, dtype=torch.bool, device=device)
            x = torch.randn(1, T, 133, device=device)
            for ts in sample_scheduler.timesteps:
                pred_x0 = denoiser(x, mask_1, ts.unsqueeze(0).to(device), text_1)
                eps = x0_to_eps(sample_scheduler, pred_x0, x, ts.item())
                x = sample_scheduler.step(eps, ts, x).prev_sample.float()

        gen_norm = x[0].cpu().numpy()   # [T, 133]
        gen_raw  = gen_norm * (std + 1e-10) + mean
        gt_raw   = norm[:T] * (std + 1e-10) + mean

        rmse = np.sqrt(np.mean((gt_raw - gen_raw[:T])**2))
        print(f"  RMSE (GT vs Generated): {rmse:.4f}")

        # 시각화
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, FFMpegWriter

            def to_joints(feat133):
                Tf = feat133.shape[0]
                pelvis   = np.zeros((Tf, 1, 3), np.float32)
                body_ric = feat133[:, 4:43].reshape(Tf, 13, 3)
                body_14  = np.concatenate([pelvis, body_ric], axis=1)
                lhand = feat133[:, 43:88].reshape(Tf, 15, 3) + body_14[:, 12:13]
                rhand = feat133[:, 88:133].reshape(Tf, 15, 3) + body_14[:, 13:14]
                return np.concatenate([body_14, lhand, rhand], axis=1)  # [T,44,3]

            gt_j  = to_joints(gt_raw)   - to_joints(gt_raw)[:, 3:4]
            gen_j = to_joints(gen_raw)  - to_joints(gen_raw)[:, 3:4]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f'1-sample overfit  RMSE={rmse:.4f}')
            for ax, label in zip([ax1, ax2], ['GT', 'Generated']):
                ax.set_title(label); ax.set_xlim(-1,1); ax.set_ylim(-1,1)
                ax.set_aspect('equal'); ax.axis('off')

            def update(f):
                for ax, jt in zip([ax1, ax2], [gt_j, gen_j]):
                    ax.cla()
                    ax.set_xlim(-1,1); ax.set_ylim(-1,1)
                    ax.set_aspect('equal'); ax.axis('off')
                    ax.scatter(jt[f,:,0], -jt[f,:,1], s=5, c='blue')
                return []

            anim = FuncAnimation(fig, update, frames=T, interval=40)
            out = 'overfit_133d_result.mp4'
            anim.save(out, writer=FFMpegWriter(fps=25, bitrate=2000))
            plt.close()
            print(f"  Video saved: {out}")
        except Exception as e:
            print(f"  Visualization skipped: {e}")

if __name__ == "__main__":
    main()