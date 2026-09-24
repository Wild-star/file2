#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — 代价体引用方式的同条件对比（论文消融表自动生成）

核心问题：在【与 corr 锚相同的尺度(1/2)、相同的窄带(R=4)、且去掉 d_cv 辅助项】的
公平条件下，带 3D 正则化聚合的代价体锚（gev）是否优于无聚合的逐点相关锚（corr）？

对比配置（同数据 / 同种子 / 同优化器 / 同步数）：
  no_anchor       : 无锚（WAFT 式纯 warp 基线）
  corr            : SparseCorrAnchor —— 无聚合逐点相关（① 匹配证据，无 ② 3D 正则化）
  gev-full3d@1/2  : GEVCostAnchor + 完整 (3,3,3) 3D 聚合，1/2 尺度、R=4（① + ②）
  gev-sep3d@1/2   : GEVCostAnchor + 可分离 3D ((1,3,3)+(3,1,1))，1/2 尺度、R=4（① + ②，更轻）

用法：
    python step3_ablation.py                 # 默认 STEPS=120 BATCH=4
    STEPS=60 BATCH=2 python step3_ablation.py
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from step2_fusion_composite import FusionWarpStereo, make_synthetic_batch, mixture_laplace_nll

STEPS = int(os.environ.get('STEPS', '120'))
BATCH = int(os.environ.get('BATCH', '4'))
H, W, MAX_DISP, SEED, LR = 96, 128, 64, 0, 2e-3

CONFIGS = [
    ('no_anchor',      dict(model=dict(anchor_kind='corr', use_anchor=False), d_cv_w=0.0)),
    ('corr',           dict(model=dict(anchor_kind='corr', use_anchor=True),  d_cv_w=0.0)),
    ('gev-full3d@1/2', dict(model=dict(anchor_kind='gev', agg_kind='full3d',
                                       gev_downsample=1, gev_R=4), d_cv_w=0.0)),
    ('gev-sep3d@1/2',  dict(model=dict(anchor_kind='gev', agg_kind='sep3d',
                                       gev_downsample=1, gev_R=4), d_cv_w=0.0)),
]


def train_one(name, cfg):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = FusionWarpStereo(img_h=H, img_w=W, **cfg['model']).train()
    n_params = sum(p.numel() for p in model.parameters())
    d_cv_w = cfg['d_cv_w']

    left, right, gt, valid = make_synthetic_batch(BATCH, H, W, MAX_DISP, seed=SEED)
    gt_half = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)

    hist = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        out = model(left, right)
        loss = 0.5 * mixture_laplace_nll((out['d_gm'] - gt_half).abs())
        if d_cv_w > 0 and out.get('d_cv') is not None:
            loss = loss + d_cv_w * mixture_laplace_nll((out['d_cv'] - gt_half).abs())
        for i, p in enumerate(out['preds']):
            w = 0.5 ** (len(out['preds']) - 1 - i)
            loss = loss + w * mixture_laplace_nll((p - gt).abs()[valid])
        loss = loss + 0.02 * torch.stack(out['gates']).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                epe = (out['preds'][-1] - gt).abs()[valid].mean().item()
                bad1 = ((out['preds'][-1] - gt).abs() > 1.0)[valid].float().mean().item()
            hist.append(dict(step=step, epe=epe, bad1=bad1, loss=loss.item()))

    dt = time.time() - t0
    res = dict(name=name, params=n_params, time_s=round(dt, 1),
               epe_first=hist[0]['epe'], epe_last=hist[-1]['epe'],
               bad1_last=hist[-1]['bad1'], steps=STEPS, hist=hist)
    print(f"  [{name:>15}] {STEPS}步 {dt:6.1f}s  EPE {hist[0]['epe']:6.3f} -> {hist[-1]['epe']:6.3f} px  "
          f"bad1px {hist[-1]['bad1']*100:5.1f}%", flush=True)
    return res


def main():
    print(f"[对比] {len(CONFIGS)} 配置 × {STEPS} 步 × batch {BATCH}（同数据/种子）\n")
    results = [train_one(name, cfg) for name, cfg in CONFIGS]

    base = results[0]['epe_last']
    print("\n" + "=" * 82)
    print("【消融表】EPE 越小越好；相对基线 (no_anchor) 的降幅")
    print("=" * 82)
    print(f"{'配置':<16}{'参数量':>9}{'EPE首':>9}{'EPE末':>9}{'bad1px':>8}{'耗时s':>8}{'相对降幅':>9}")
    print("-" * 82)
    for r in results:
        rel = (1 - r['epe_last'] / base) * 100
        print(f"{r['name']:<16}{r['params']/1e3:>7.1f}K{r['epe_first']:>9.3f}{r['epe_last']:>9.3f}"
              f"{r['bad1_last']*100:>7.1f}%{r['time_s']:>8.1f}{rel:>8.1f}%")
    print("=" * 82)

    with open('step3_ablation.json', 'w') as f:
        json.dump(dict(steps=STEPS, batch=BATCH, results=results), f, indent=2)
    print("[输出] 已存 step3_ablation.json")


if __name__ == '__main__':
    main()
