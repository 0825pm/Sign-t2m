#!/bin/bash
# ============================================================
# Sign-t2m → 원본 Light-T2M 정렬 패치
#
# 변경 1: sign_denoiser.py — cross-attn 제거, t_hidden_proj 제거
# 변경 2: sign_t2m.py — sample_motion 추론 로직 원본 방식으로
#
# 사용법:
#   cd ~/Projects/research/sign-t2m
#   bash align_to_original.sh
# ============================================================
set -e

FILE1="src/models/nets/sign_denoiser.py"
FILE2="src/models/sign_t2m.py"

echo "=== 1/2: sign_denoiser.py 수정 ==="

# --- (a) MixedModule: num_heads 파라미터 제거 ---
sed -i 's/def __init__(self, model_dim, build_mamba_block_fn, patch_size=8, mask_padding=True, num_heads=4):/def __init__(self, model_dim, build_mamba_block_fn, patch_size=8, mask_padding=True):/' "$FILE1"
echo "  ✅ MixedModule.__init__: num_heads 파라미터 제거"

# --- (b) MixedModule: cross_attn, cross_norm 정의 제거 ---
sed -i '/# Cross-attention: motion frames attend to text tokens/,/self.cross_norm = nn.LayerNorm(model_dim)/d' "$FILE1"
echo "  ✅ MixedModule.__init__: cross_attn, cross_norm 정의 삭제"

# --- (c) MixedModule.forward: text_hidden 파라미터 제거 ---
sed -i 's/def forward(self, x, x_mask, y, y_mask, text_hidden=None, text_mask=None):/def forward(self, x, x_mask, y, y_mask):/' "$FILE1"
echo "  ✅ MixedModule.forward: text_hidden 파라미터 제거"

# --- (d) MixedModule.forward: cross-attn 호출 블록 제거 ---
sed -i '/# Cross-attention: downsampled motion queries/,/nx1 = self.cross_norm(nx1 + attn_out)/d' "$FILE1"
echo "  ✅ MixedModule.forward: cross-attn 호출 블록 삭제"

# --- (e) StageBlock.forward: text_hidden 전달 제거 ---
sed -i 's/def forward(self, x, x_mask, y, y_mask, text_hidden=None, text_mask=None):/def forward(self, x, x_mask, y, y_mask):/' "$FILE1"
sed -i 's/x, _ = self.mixed_module(x, x_mask, y_, y_mask, text_hidden=text_hidden, text_mask=text_mask)/x, _ = self.mixed_module(x, x_mask, y_, y_mask)/' "$FILE1"
echo "  ✅ StageBlock.forward: text_hidden 전달 제거"

# --- (f) SignDenoiser.__init__: t_hidden_proj 제거 ---
sed -i '/self.t_hidden_proj = nn.Linear(text_dim, base_dim)/d' "$FILE1"
echo "  ✅ SignDenoiser.__init__: t_hidden_proj 삭제"

# --- (g) SignDenoiser.forward: text_hidden 관련 코드 제거 ---
sed -i '/# Token-level text hidden states for cross-attention/d' "$FILE1"
sed -i '/text_hidden = self.t_hidden_proj/d' "$FILE1"
sed -i '/text_hidden_mask = text\["mask"\]/d' "$FILE1"
sed -i 's/x, text_feat = layer(x, x_mask, text_feat, text_mask,$/x, text_feat = layer(x, x_mask, text_feat, text_mask)/' "$FILE1"
sed -i '/text_hidden=text_hidden, text_mask=text_hidden_mask)/d' "$FILE1"
echo "  ✅ SignDenoiser.forward: text_hidden 관련 코드 삭제"

echo ""
echo "=== 2/2: sign_t2m.py 수정 ==="

# --- sample_motion: prediction_type=sample 시 원본 방식 (x0→eps 변환 후 step) ---
# 현재: x0 공간에서 직접 CFG → scheduler.step(pred_x0, ...)
# 원본: x0 → eps 변환 → CFG in eps space → scheduler.step(pred_noise, ...)

python3 << 'PYEOF'
import re

with open("src/models/sign_t2m.py", "r") as f:
    code = f.read()

# Replace the sample prediction block in sample_motion
old_sample_block = '''            elif prediction_type == "sample":
                cond_x0, uncond_x0 = output.chunk(2)
                # CFG directly in x0 space
                pred_x0 = uncond_x0 + self.hparams.guidance_scale * (cond_x0 - uncond_x0)
                pred_motion = self.sample_scheduler.step(
                    pred_x0, t, pred_motion
                ).prev_sample.float()'''

new_sample_block = '''            elif prediction_type == "sample":
                cond_x0, uncond_x0 = output.chunk(2)
                cond_eps, uncond_eps = self._obtain_eps_from_x0(cond_x0, uncond_x0, t, pred_motion)
                pred_noise = uncond_eps + self.hparams.guidance_scale * (cond_eps - uncond_eps)
                pred_motion = self.sample_scheduler.step(
                    pred_noise, t, pred_motion
                ).prev_sample.float()'''

if old_sample_block in code:
    code = code.replace(old_sample_block, new_sample_block)
    print("  ✅ sample_motion: sample prediction → 원본 방식 (x0→eps→CFG)")
else:
    print("  ⚠️ sample_motion 블록을 찾지 못함. 수동 확인 필요.")

# Also fix the assertion to allow both epsilon and sample
old_assert = '''assert pt == "epsilon"'''
new_assert = '''assert pt in ("epsilon", "sample")'''
if old_assert in code:
    code = code.replace(old_assert, new_assert)
    print("  ✅ prediction_type assertion: sample도 허용")

with open("src/models/sign_t2m.py", "w") as f:
    f.write(code)
PYEOF

echo ""
echo "=== 완료! ==="
echo ""
echo "변경 확인:"
echo "  git diff src/models/nets/sign_denoiser.py"
echo "  git diff src/models/sign_t2m.py"
echo ""
echo "커밋:"
echo "  git add -A"
echo "  git commit -m 'align: remove cross-attn, fix sample inference to match Light-T2M original'"
