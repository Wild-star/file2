#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 9 — 三个差异点的多种子消融（论文核心贡献的稳健验证）

step4 的单种子对照显示 token 稀疏贡献最大（fusion 2.050 → fusion_nosparse 2.688）、锚次之。
但 step5.5 已证明单种子不可靠，必须多种子复验。本脚本在 easy 数据上、3 种子 × 3 配置：
  fusion          : 全模块（GlobalMatcher + token 稀疏 + 门控 + 窄带相关锚）
  fusion_nosparse : 关 token 稀疏（use_sparse=False）  —— 测「token 稀疏」贡献
  fusion_noanchor : 关相关锚（use_anchor=False）       —— 测「锚」贡献

判定：若 mean(fusion) 稳健优于 mean(fusion_nosparse) 且 std 不重叠，则 token 稀疏增益稳健。
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from step2_fusion_composite import FusionWarpStereo, mixture_laplace_nll
from step7_difficulty import make_difficulty_batch

STEPS = int(os.environ.get('STEPS', '80'))
BATCH = int(os.environ.get('BATCH', '4'))
H, W, MAX_DISP, LR = 96, 128, 64, 2e-3
SEEDS = [0, 1, 2]


def train_one(build_fn, data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_fn().train()
    left, right, gt, valid = data
    gt_half = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)

    for step in range(1, STEPS + 1):
        out = model(left, right)
        loss = 0.5 * mixture_laplace_nll((out['d_gm'] - gt_half).abs())
        for i, p in enumerate(out['preds']):
            w = 0.5 ** (len(out['preds']) - 1 - i)
            loss = loss + w * mixture_laplace_nll((p - gt).abs()[valid])
        loss = loss + 0.02 * torch.stack(out['gates']).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        epe = (model(left, right)['preds'][-1] - gt).abs()[valid].mean().item()
    return epe


def main():
    easy = make_difficulty_batch(BATCH, H, W, MAX_DISP, seed=0, texture='smooth', occlusion=False)
    configs = [
        ('fusion',          lambda: FusionWarpStereo(img_h=H, img_w=W, iters=2)),
        ('fusion_nosparse', lambda: FusionWarpStereo(img_h=H, img_w=W, iters=2, use_sparse=False)),
        ('fusion_noanchor', lambda: FusionWarpStereo(img_h=H, img_w=W, iters=2, use_anchor=False)),
    ]
    print(f"[多种子消融] {SEEDS} 种子 × {len(configs)} 配置（easy 数据，固定 seed=0）\n")

    results = {}
    for name, build_fn in configs:
        epes = []
        for s in SEEDS:
            e = train_one(build_fn, easy, s)
            epes.append(e)
            print(f"  [{name} seed{s}] EPE {e:.3f}", flush=True)
        arr = np.array(epes)
        results[name] = dict(mean=arr.mean(), std=arr.std(), seeds=epes)

    base = results['fusion']['mean']
    print("\n" + "=" * 72)
    print("【多种子消融】mean±std；Δ = 去掉该模块后的 EPE 增量（>0 说明该模块有益）")
    print("=" * 72)
    print(f"{'配置':<18}{'EPE mean±std':>16}{'Δ vs fusion':>13}")
    print("-" * 72)
    for name in ['fusion', 'fusion_nosparse', 'fusion_noanchor']:
        r = results[name]
        delta = r['mean'] - base
        tag = '（baseline）' if name == 'fusion' else ''
        print(f"{name:<18}{r['mean']:>8.3f}±{r['std']:.3f}{delta:>+11.3f}{tag}")
    print("=" * 72)
    print("解读：fusion_nosparse 的 Δ>0 且 std 不重叠 → token 稀疏稳健有益；")
    print("      fusion_noanchor 的 Δ>0 且 std 不重叠 → 锚稳健有益。")

    with open('step9_multiseed_ablation.json', 'w') as f:
        json.dump(dict(steps=STEPS, batch=BATCH, seeds=SEEDS, results=results), f, indent=2)
    print("[输出] 已存 step9_multiseed_ablation.json")


if __name__ == '__main__':
    main()
