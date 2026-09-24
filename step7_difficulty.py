#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 7 — 难度相变验证：OT / 不确定性引导是否只在「困难场景」才产生增益

核心假设（深度思考后）：前两轮光流模块（OT 全局匹配、逐像素不确定性引导）在合成分上
负结果，是因为合成分「太简单」（无遮挡、无重复纹理、视差平滑）——这些高级机制的
价值场景（歧义匹配、不均匀误差、遮挡边界）不存在。若把数据难度抬升（周期重复纹理 +
前景块视差跳变 + 遮挡带），这些机制应「转正」。

本脚本实现难度可控的数据生成器，并在 easy / hard 两档难度上扫 3 个模型：
  fusion    : softmax 初始化 + 固定系数 mixture-Laplace（baseline）
  fusion_ot : Sinkhorn OT 初始化 + 置信度引导（FlowIt）
  fusion_unc: 逐像素 σ 引导损失（U²Flow）

判定：若 OT / σ 在 hard 上相对 fusion 的差距显著收窄或转正，则「难度相变」假设成立。

用法：
    python step7_difficulty.py            # STEPS=120 BATCH=4
    STEPS=80 python step7_difficulty.py
"""
import os, sys, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from step2_fusion_composite import FusionWarpStereo, mixture_laplace_nll, disp_warp, make_synthetic_batch
from step5_uncertainty import UncertaintyFusionWarp, uncertainty_nll
from step6_ot import OTFusionWarp

STEPS = int(os.environ.get('STEPS', '120'))
BATCH = int(os.environ.get('BATCH', '4'))
H, W, MAX_DISP, SEED, LR = 96, 128, 64, 0, 2e-3


# --------------------------------------------------------------------------- #
# 难度可控数据生成器
# --------------------------------------------------------------------------- #
def make_difficulty_batch(B, H, W, max_disp=64, seed=0, texture='smooth', occlusion=False):
    """texture: 'smooth'（低纹理多斑点）| 'repetitive'（周期纹理→歧义匹配）；
    occlusion: 是否加入前景块视差跳变 + 遮挡带。"""
    g = torch.Generator().manual_seed(seed)
    lefts, disps, valids = [], [], []
    for _ in range(B):
        # ---- 纹理 ----
        if texture == 'repetitive':
            yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                    torch.arange(W, dtype=torch.float32), indexing='ij')
            ch = []
            for k, (fx, fy, ph) in enumerate([(5, 3, 0.0), (3, 5, 1.3), (4, 4, 2.6)]):
                t = 0.5 * torch.sin(fx * 2 * math.pi * xx / W + ph) * \
                    torch.sin(fy * 2 * math.pi * yy / H + ph)
                ch.append(t)
            left = torch.stack(ch, 0).unsqueeze(0)                 # (1,3,H,W)
            left = left * 0.35 + 0.5 + torch.randn(1, 3, H, W, generator=g) * 0.04
            left = left.clamp(0, 1)
        else:
            z = torch.randn(1, 3, H // 8, W // 8, generator=g)
            tex = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=True)
            left = torch.sigmoid(tex * 1.5) * 0.7 + 0.3 + torch.randn(1, 3, H, W, generator=g) * 0.06
            left = left.clamp(0, 1)

        # ---- 平滑背景视差场 ----
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing='ij')
        d0 = 4 + torch.rand(1, generator=g).item() * max_disp * 0.4
        gx = (torch.rand(1, generator=g).item() - 0.5) * max_disp * 0.6
        gy = (torch.rand(1, generator=g).item() - 0.5) * max_disp * 0.6
        amp = torch.rand(1, generator=g).item() * 6
        fx = 2 + torch.rand(1, generator=g).item() * 4
        disp = d0 + gx * (xx / W) + gy * (yy / H) + amp * torch.sin(fx * 2 * math.pi * xx / W)
        disp = disp.clamp(1, max_disp)
        valid = torch.ones(H, W, dtype=torch.bool)

        # ---- 前景块视差跳变 + 遮挡带 ----
        if occlusion:
            # 随机前景矩形（更近 → 视差更大）
            bx = int(torch.randint(4, W // 3, (1,), generator=g).item())
            bw = int(torch.randint(W // 6, W // 3, (1,), generator=g).item())
            by = int(torch.randint(4, H // 3, (1,), generator=g).item())
            bh = int(torch.randint(H // 6, H // 3, (1,), generator=g).item())
            x0, x1 = bx, min(W, bx + bw)
            y0, y1 = by, min(H, by + bh)
            d_fg = disp[y0:y1, x0:x1].mean() + 18                  # 前景视差跳变
            disp[y0:y1, x0:x1] = d_fg
            # 遮挡带：前景块左边缘的背景，在右图被前景挡住（无匹配）
            occ_w = int(min(14, (d_fg - disp[y0:y1, max(0, x0 - 14):x0].mean()).clamp(1, 20)))
            valid[y0:y1, max(0, x0 - occ_w):x0] = False

        lefts.append(left)
        disps.append(disp.unsqueeze(0).unsqueeze(0))
        valids.append(valid.unsqueeze(0).unsqueeze(0))

    left = torch.cat(lefts, 0)
    gt = torch.cat(disps, 0)
    valid = torch.cat(valids, 0).bool()
    right = disp_warp(left, gt, padding_mode='border')
    return left, right, gt, valid


# --------------------------------------------------------------------------- #
# 训练 + 扫描
# --------------------------------------------------------------------------- #
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


def train_one(name, model, kind, data):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model.train()
    left, right, gt, valid = data
    gt_half = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)

    t0 = time.time()
    epe_first = epe_last = float('nan')
    for step in range(1, STEPS + 1):
        out = model(left, right)
        loss = loss_of(kind, out, gt, gt_half, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                epe = (out['preds'][-1] - gt).abs()[valid].mean().item()
            if step == 1:
                epe_first = epe
            epe_last = epe

    dt = time.time() - t0
    print(f"  [{name:>12}] {STEPS}步 {dt:6.1f}s  EPE {epe_first:6.3f} -> {epe_last:6.3f} px", flush=True)
    return dict(name=name, epe_first=epe_first, epe_last=epe_last, time_s=round(dt, 1))


def main():
    easy = make_difficulty_batch(BATCH, H, W, MAX_DISP, seed=SEED, texture='smooth', occlusion=False)
    hard = make_difficulty_batch(BATCH, H, W, MAX_DISP, seed=SEED, texture='repetitive', occlusion=True)
    print(f"[难度扫描] easy(平滑纹理,无遮挡) vs hard(重复纹理+遮挡) × 3 模型\n")

    configs = [
        ('easy_fusion', lambda: FusionWarpStereo(img_h=H, img_w=W, iters=2), 'plain', easy),
        ('easy_ot',     lambda: OTFusionWarp(img_h=H, img_w=W, iters=2), 'conf', easy),
        ('easy_unc',    lambda: UncertaintyFusionWarp(img_h=H, img_w=W, iters=2), 'unc', easy),
        ('hard_fusion', lambda: FusionWarpStereo(img_h=H, img_w=W, iters=2), 'plain', hard),
        ('hard_ot',     lambda: OTFusionWarp(img_h=H, img_w=W, iters=2), 'conf', hard),
        ('hard_unc',    lambda: UncertaintyFusionWarp(img_h=H, img_w=W, iters=2), 'unc', hard),
    ]
    results = {}
    for name, fn, kind, data in configs:
        results[name] = train_one(name, fn(), kind, data)

    def rel(x, base): return (1 - x['epe_last'] / base['epe_last']) * 100

    print("\n" + "=" * 82)
    print("【难度扫描】EPE 越小越好；「相对 fusion 的降幅」= 模块在对应难度下的增益")
    print("=" * 82)
    print(f"{'难度':<6}{'模型':<10}{'EPE首':>9}{'EPE末':>9}{'相对fusion':>11}")
    print("-" * 82)
    for diff in ['easy', 'hard']:
        base = results[f'{diff}_fusion']
        for m in ['fusion', 'ot', 'unc']:
            r = results[f'{diff}_{m}']
            tag = '（baseline）' if m == 'fusion' else ''
            print(f"{diff:<6}{m:<10}{r['epe_first']:>9.3f}{r['epe_last']:>9.3f}{rel(r, base):>10.1f}%{tag}")
    print("=" * 82)
    print("判定：若 OT/σ 在 hard 上的相对降幅 > easy 上的相对降幅（尤其由负转正），则「难度相变」假设成立。")

    with open('step7_difficulty.json', 'w') as f:
        json.dump(dict(steps=STEPS, batch=BATCH, results=results), f, indent=2)
    print("[输出] 已存 step7_difficulty.json")


if __name__ == '__main__':
    main()
