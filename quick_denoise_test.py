import torch
torch.backends.cudnn.enabled = False
import os, sys, numpy as np
sys.path.insert(0, '.')
from src.models.nets.sign_denoiser import SignDenoiser
from src.models.nets.text_encoder import CLIP
from diffusers import DDPMScheduler, UniPCMultistepScheduler
import torch.nn.functional as F

device = torch.device('cuda:0')
DATA = '/home/user/Projects/research/SOKE/data/How2Sign_133d'

# Load data
npy_dir = f'{DATA}/train/poses'
npy_file = sorted(os.listdir(npy_dir))[0]
raw = np.load(os.path.join(npy_dir, npy_file))
mean = torch.load(f'{DATA}/mean_133.pt', map_location='cpu').numpy()
std = torch.load(f'{DATA}/std_133.pt', map_location='cpu').numpy()
norm = (raw - mean) / (std + 1e-10)
T = (min(norm.shape[0], 100) // 4) * 4
motion = torch.from_numpy(norm[:T]).float().unsqueeze(0).to(device)

# Build
text_encoder = CLIP().to(device).eval()
denoiser = SignDenoiser(motion_dim=133, max_motion_len=401, text_dim=512, stage_dim="256*4").to(device).train()
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2", clip_sample=False, prediction_type="epsilon")

# Overfit (same as before)
B = 32
motion_b = motion.repeat(B, 1, 1)
optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-3)
with torch.no_grad():
    text_dict = text_encoder(["a person signing"] * B, device)
mask = torch.ones(B, T, dtype=torch.bool, device=device)

print("=== Overfit 10000 steps ===")
for step in range(10000):
    optimizer.zero_grad()
    t = torch.randint(0, 1000, (B,), device=device)
    noise = torch.randn_like(motion_b)
    x_t = scheduler.add_noise(motion_b, noise, t)
    pred = denoiser(x_t, mask, t, text_dict)
    loss = F.mse_loss(pred, noise)
    loss.backward()
    optimizer.step()
    if step % 2000 == 0 or step == 9999:
        print(f"  step {step:5d}  loss={loss.item():.6f}")

# Now test denoising from different noise levels
denoiser.eval()
with torch.no_grad():
    text_1 = text_encoder(["a person signing"], device)
    mask_1 = torch.ones(1, T, dtype=torch.bool, device=device)

    for t_val in [50, 200, 500, 999]:
        t = torch.tensor([t_val], device=device)
        noise = torch.randn_like(motion)
        x_t = scheduler.add_noise(motion, noise, t)

        # Single-step denoising (predict noise, subtract)
        pred_noise = denoiser(x_t, mask_1, t, text_1)
        alpha = scheduler.alphas_cumprod[t_val]
        pred_x0 = (x_t - (1-alpha)**0.5 * pred_noise) / alpha**0.5

        rmse = ((pred_x0 - motion)**2).mean().sqrt().item()
        print(f"  t={t_val:4d}: single-step recon RMSE={rmse:.4f}")

print("\nIf low-t RMSE is good but high-t is bad => need more overfit steps")
print("If all RMSE are bad => model not learning properly")
