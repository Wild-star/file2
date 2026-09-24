#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FusionWarp 训练 smoke test —— 用最少数据验证「训练循环能跑通」。

不是训练，只是端到端冒烟：构建 fusion 模型 → criterion → optimizer →
随机小 batch → 跑 N 步 forward/loss/backward/step，确认不报错且 loss 有梯度。

用法：
    python train_smoke.py                 # corr 锚，5 步
    STEPS=10 ANCHOR=gev python train_smoke.py
    STEPS=5 GLOBAL_INIT=1 python train_smoke.py   # 加 GlobalMatcher 初始视差
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bridgedepth.config import get_cfg
from algorithms.waft import WAFT
from bridgedepth.loss import build_criterion

STEPS = int(os.environ.get('STEPS', '5'))
ANCHOR = os.environ.get('ANCHOR', 'corr')          # corr | gev
GLOBAL_INIT = os.environ.get('GLOBAL_INIT', '0') == '1'
H, W = 128, 160
MAX_DISP = 128
BATCH = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[设备] {device}  |  STEPS={STEPS}  ANCHOR={ANCHOR}  GLOBAL_INIT={GLOBAL_INIT}  input={H}x{W}")

# ---- 构建 fusion 配置 ----
cfg = get_cfg()
cfg.merge_from_file("configs/SynLarge/DAv2S-4.yaml")
cfg.WAFT.MAX_DISP = MAX_DISP
cfg.WAFT.FUSION.ENABLED = True
cfg.WAFT.FUSION.USE_ANCHOR = True
cfg.WAFT.FUSION.ANCHOR_KIND = ANCHOR
cfg.WAFT.FUSION.USE_GLOBAL_INIT = GLOBAL_INIT
cfg.freeze()

model = WAFT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[模型] 总参数 {n_params/1e6:.2f}M（可训练 {n_tr/1e6:.2f}M）")

criterion = build_criterion(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=1e-5)

# ---- 最少数据：随机合成立体对 + 随机视差 GT ----
torch.manual_seed(0)
sample = {
    "img1": torch.rand(BATCH, 3, H, W, device=device) * 255,
    "img2": torch.rand(BATCH, 3, H, W, device=device) * 255,
    "disp": torch.rand(BATCH, H, W, device=device) * (MAX_DISP * 0.5),   # GT 视差（全分辨率）
    "valid": torch.ones(BATCH, H, W, device=device),                       # 有效 mask
}

print(f"\n[训练 smoke] {STEPS} 步（前向 + loss + 反向 + step）\n")
model.train()
for step in range(1, STEPS + 1):
    result_dict = model(sample)
    loss_dict, metrics = criterion(result_dict, sample, log=True)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    for param in model.parameters():
        param.grad = None
    losses.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"  step {step}/{STEPS}  loss={losses.item():.4f}  "
          f"EPE={metrics['EPE'].item():.3f}px  mixlap={loss_dict['mixlap'].item():.4f}  "
          f"init={loss_dict['init'].item():.4f}", flush=True)

print(f"\n[结果] ✅ 训练循环 {STEPS} 步跑通（无报错，loss 有梯度）")
print(f"       loss 首末：{None if False else '见上'}；EPE 末值见上")
