#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 8 — 多种子稳健性重测（修复单种子不可复现的方法学缺陷）

问题：之前所有对照（step3~step7）的模型在 train_one 外构造，torch.manual_seed 在内部才
调用 → 模型初始权重每次运行不同 → 单种子结论不可复现（step6 OT=-48% vs step7 easy OT=+25%）。

修复：① 数据用固定 seed 生成（same data across seeds）；② 模型在 train_one 内、
torch.manual_seed(seed) 之后构造（可复现）；③ 每个模型跑 SEEDS 个种子，报告 mean±std。

对照：fusion(softmax 初始化) vs ot(Sinkhorn OT) vs unc(σ 引导)，在 easy / hard 两档难度。
"""
import os, sys, time, json, math
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from step2_fusion_composite import FusionWarpStereo, mixture_laplace_nll, disp_warp
from step5_uncertainty import UncertaintyFusionWarp, uncertainty_nll
from step6_ot import OTFusionWarp
from step7_difficulty import make_difficulty_batch

STEPS = int(os.environ.get('STEPS', '80'))
BATCH = int(os.environ.get('BATCH', '4'))
H, W, MAX_DISP, LR = 96, 128, 64, 2e-3
SEEDS = [0, 1, 2]


def loss_of(kind, out, gt, gt_half, valid):
    loss = 0.0
    if out.get('d_gm') is not None:
        err = (out['d_gm'] - gt_half).abs()
        if kind == 'conf' and out.get('conf') is not None:
            loss = loss + 0.5 * mixture_laplace_nll(err) + 0.5 * (err * out['conf']).mean()
        else:
            loss = loss + 0.5 * mixture_laplace_nll(err)
    for i, p in enumerate(out['preds']):
        w = 0.5 ** (len(out['preds']) - 1 - i)
        err = (p - gt).abs()[valid]
        if kind == 'unc' and out.get('sigmas'):
            loss = loss + w * uncertainty_nll(err, out['sigmas'][i][valid])
        else:
            loss = loss + w * mixture_laplace_nll(err)
    if out.get('gates'):
        loss = loss + 0.02 * torch.stack(out['gates']).mean()
    return loss


def train_one(build_fn, kind, data, seed):
    torch.manual_seed(seed)               # 构造前重置 → 可复现
    np.random.seed(seed)
    model = build_fn().train()            # 关键修复：在 seed 内构造
    left, right, gt, valid = data
    gt_half = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)

    for step in range(1, STEPS + 1):
        out = model(left, right)
        loss = loss_of(kind, out, gt, gt_half, valid)
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
    hard = make_difficulty_batch(BATCH, H, W, MAX_DISP, seed=0, texture='repetitive', occlusion=True)

    models = [
        ('fusion', lambda: FusionWarpStereo(img_h=H, img_w=W, iters=2), 'plain'),
        ('ot',     lambda: OTFusionWarp(img_h=H, img_w=W, iters=2), 'conf'),
        ('unc',    lambda: UncertaintyFusionWarp(img_h=H, img_w=W, iters=2), 'unc'),
    ]
    print(f"[多种子] {SEEDS} 种子 × {len(models)} 模型 × 2 难度（数据固定 seed=0）\n")

    results = {}
    for diff, data in [('easy', easy), ('hard', hard)]:
        for mname, build_fn, kind in models:
            epes = []
            for s in SEEDS:
                e = train_one(build_fn, kind, data, s)
                epes.append(e)
                print(f"  [{diff}_{mname} seed{s}] EPE {e:.3f}", flush=True)
            arr = np.array(epes)
            results[f'{diff}_{mname}'] = dict(mean=arr.mean(), std=arr.std(), seeds=epes)

    print("\n" + "=" * 76)
    print("【多种子结果】mean±std；相对 fusion 的降幅用 mean 计算")
    print("=" * 76)
    print(f"{'难度':<6}{'模型':<8}{'EPE mean±std':>16}{'相对fusion':>11}")
    print("-" * 76)
    for diff in ['easy', 'hard']:
        base = results[f'{diff}_fusion']['mean']
        for m in ['fusion', 'ot', 'unc']:
            r = results[f'{diff}_{m}']
            rel = (1 - r['mean'] / base) * 100
            tag = '（baseline）' if m == 'fusion' else ''
            print(f"{diff:<6}{m:<8}{r['mean']:>8.3f}±{r['std']:.3f}{rel:>10.1f}%{tag}")
    print("=" * 76)

    with open('step8_multiseed.json', 'w') as f:
        json.dump(dict(steps=STEPS, batch=BATCH, seeds=SEEDS, results=results), f, indent=2)
    print("[输出] 已存 step8_multiseed.json")


if __name__ == '__main__':
    main()
