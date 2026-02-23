import torch
torch.backends.cudnn.enabled = False

import os, sys, numpy as np
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))

from src.models.nets.sign_denoiser import SignDenoiser
from src.models.nets.text_encoder import CLIP
from diffusers import DDPMScheduler

def main():
    device = torch.device('cuda:0')
    DATA = '/home/user/Projects/research/SOKE/data/How2Sign_133d'
    npy_dir = f'{DATA}/train/poses'
    npy_file = sorted(os.listdir(npy_dir))[0]
    raw = np.load(os.path.join(npy_dir, npy_file))
    mean = torch.load(f'{DATA}/mean_133.pt', map_location='cpu').numpy()
    std = torch.load(f'{DATA}/std_133.pt', map_location='cpu').numpy()
    norm = (raw - mean) / (std + 1e-10)
    T = (min(norm.shape[0], 100) // 4) * 4
    motion_1 = torch.from_numpy(norm[:T]).float().unsqueeze(0).to(device)

    # Repeat same sample 32 times — same motion, different timesteps
    B = 32
    motion = motion_1.repeat(B, 1, 1)  # [32, T, 133]
    print(f"Sample: {npy_file}, T={T}, batch={B}")

    text_encoder = CLIP().to(device).eval()
    for p in text_encoder.parameters(): p.requires_grad_(False)
    denoiser = SignDenoiser(motion_dim=133, max_motion_len=401, text_dim=512, stage_dim="256*4").to(device).train()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2", clip_sample=False, prediction_type="epsilon")
    print(f"Params: {sum(p.numel() for p in denoiser.parameters() if p.requires_grad)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-3)
    with torch.no_grad():
        text_dict = text_encoder(["a person signing"] * B, device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    print("\n=== 1 sample x 32 timesteps, lr=1e-3, 5000 steps ===\n")
    for step in range(5000):
        optimizer.zero_grad()
        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn_like(motion)
        x_t = scheduler.add_noise(motion, noise, t)
        pred = denoiser(x_t, mask, t, text_dict)
        loss = F.mse_loss(pred, noise)
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == 4999:
            print(f"  step {step:5d}  loss={loss.item():.6f}")

    f = loss.item()
    print(f"\n{'PASS' if f < 0.1 else 'PARTIAL' if f < 0.5 else 'FAIL'} — loss={f:.6f}")

if __name__ == "__main__":
    main()
