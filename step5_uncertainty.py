#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5 — 光流领域模块融合：逐像素不确定性引导（U²Flow, CVPR 2026 Oral）

策略（按用户指示）：优先融合**非双目立体匹配领域**的创新模块——跨域迁移本身即可作为
创新性说明，不涉及与双目领域成果的撞车。本脚本把光流领域 U²Flow 的「逐像素
aleatoric 不确定性引导损失」融合进 FusionWarp 的双目迭代框架：

  baseline  fusion     : 固定系数 mixture-of-Laplace（SEA-RAFT）
  融合后    fusion_unc : 迭代解码器额外预测逐像素不确定度 σ，
                         损失 L = |e|/σ + log σ（U²Flow / Kendall&Gal 2018）

创新性说明（跨域迁移）：光流的逐像素不确定性引导损失首次被引入「相关 + warp 互补」
的双目迭代匹配框架，替代固定系数的鲁棒损失，使难例（遮挡/无纹理）自动获得自适应
降权，且不确定度图可作为免费的可视化/诊断信号。

用法：
    python step5_uncertainty.py            # STEPS=120 BATCH=4 对照
    STEPS=80 python step5_uncertainty.py
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from step2_fusion_composite import (
    FeatureEncoder, MatchFeat, GlobalMatcher, SparseCorrAnchor, GatedFusion,
    TokenSparseViT, disp_warp, convex_upsample, mixture_laplace_nll, make_synthetic_batch,
)

STEPS = int(os.environ.get('STEPS', '120'))
BATCH = int(os.environ.get('BATCH', '4'))
H, W, MAX_DISP, SEED, LR = 96, 128, 64, 0, 2e-3


class UncertaintyFusionWarp(nn.Module):
    """FusionWarp + 逐像素不确定性头：σ=softplus(conf_head(net))，用于不确定性引导损失。"""
    def __init__(self, C=32, hidden=32, G=8, R=4, D_bins=9, iters=2, img_h=96, img_w=128):
        super().__init__()
        self.encoder = FeatureEncoder(C)
        self.match = MatchFeat(C)
        self.global_matcher = GlobalMatcher(C, D_bins, img_h=img_h, img_w=img_w)
        self.fusion = GatedFusion(C)
        self.delta_proj = nn.Conv2d(4 * C + 1, hidden, 3, padding=1)
        self.delta_decoder = TokenSparseViT(hidden, depth=2, heads=4, patch=8, out_ch=hidden, sparse=True)
        self.disp_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.mask_head = nn.Conv2d(hidden, 9 * 4, 3, padding=1)
        self.conf_head = nn.Conv2d(hidden, 1, 3, padding=1)      # 新增：不确定性头（log-σ）
        self.anchor = SparseCorrAnchor(C, G, R, hidden)
        self.iters = iters

    def forward(self, img1, img2):
        f1, f2 = self.encoder(img1), self.encoder(img2)
        m1, m2 = self.match(img1), self.match(img2)
        d_gm, g_feat = self.global_matcher(f1, f2)
        disp = d_gm
        net = None
        preds, gates, sigmas = [], [], []
        for _ in range(self.iters):
            disp = disp.detach()
            anchor = self.anchor(m1, m2, disp)
            warped = disp_warp(f2, disp, padding_mode='zeros')
            fused = self.fusion(anchor, g_feat)
            if net is None:
                net = torch.zeros_like(f1)
            x = torch.cat([f1, warped, net, disp, fused], dim=1)
            net = self.delta_proj(x)
            net, gate = self.delta_decoder(net)
            gates.append(gate)
            sigma = F.softplus(self.conf_head(net)) + 1e-3   # (B,1,H/2,W/2) 正不确定度
            sigma = F.interpolate(sigma, scale_factor=2, mode='bilinear', align_corners=True)
            sigmas.append(sigma)                              # 对齐全分辨率 GT
            delta_disp = self.disp_head(net)
            mask = 0.25 * self.mask_head(net)
            disp = disp + delta_disp
            preds.append(convex_upsample(disp * 2, mask))
        return {'preds': preds, 'gates': gates, 'd_gm': d_gm, 'd_cv': None, 'sigmas': sigmas}


def uncertainty_nll(err, sigma):
    """U²Flow / aleatoric uncertainty：L = |e|/σ + log σ（逐像素，然后 mean）。"""
    return (err / sigma + torch.log(sigma)).mean()


def train_one(name, model, use_unc):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())

    left, right, gt, valid = make_synthetic_batch(BATCH, H, W, MAX_DISP, seed=SEED)
    gt_half = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)

    hist = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        out = model(left, right)
        loss = 0.0
        if out.get('d_gm') is not None:
            loss = loss + 0.5 * mixture_laplace_nll((out['d_gm'] - gt_half).abs())
        for i, p in enumerate(out['preds']):
            w = 0.5 ** (len(out['preds']) - 1 - i)
            err = (p - gt).abs()[valid]
            if use_unc and out.get('sigmas'):
                loss = loss + w * uncertainty_nll(err, out['sigmas'][i][valid])
            else:
                loss = loss + w * mixture_laplace_nll(err)
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
                smean = out['sigmas'][-1].mean().item() if out.get('sigmas') else float('nan')
            hist.append(dict(step=step, epe=epe, bad1=bad1, loss=loss.item(), sigma_mean=smean))

    dt = time.time() - t0
    res = dict(name=name, params=n_params, time_s=round(dt, 1),
               epe_first=hist[0]['epe'], epe_last=hist[-1]['epe'],
               bad1_last=hist[-1]['bad1'], sigma_mean_last=hist[-1]['sigma_mean'],
               steps=STEPS, hist=hist)
    print(f"  [{name:>16}] {STEPS}步 {dt:6.1f}s  EPE {hist[0]['epe']:6.3f} -> {hist[-1]['epe']:6.3f} px  "
          f"bad1px {hist[-1]['bad1']*100:5.1f}%  σ̄={hist[-1]['sigma_mean']:.3f}", flush=True)
    return res


def main():
    configs = [
        ('fusion',          lambda: (UncertaintyFusionWarp(img_h=H, img_w=W, iters=2), False)),
        ('fusion_unc',      lambda: (UncertaintyFusionWarp(img_h=H, img_w=W, iters=2), True)),
    ]
    print(f"[对照] 固定系数混合拉普拉斯 vs 逐像素不确定性引导（同架构/种子）\n")
    results = []
    for name, fn in configs:
        model, use_unc = fn()
        results.append(train_one(name, model, use_unc))

    base = results[0]['epe_last']
    print("\n" + "=" * 78)
    print("【对照】EPE 越小越好；相对 baseline (固定系数) 的降幅")
    print("=" * 78)
    print(f"{'配置':<14}{'参数量':>9}{'EPE首':>9}{'EPE末':>9}{'bad1px':>8}{'σ̄末':>8}{'相对降幅':>9}")
    print("-" * 78)
    for r in results:
        rel = (1 - r['epe_last'] / base) * 100
        print(f"{r['name']:<14}{r['params']/1e3:>7.1f}K{r['epe_first']:>9.3f}{r['epe_last']:>9.3f}"
              f"{r['bad1_last']*100:>7.1f}%{r['sigma_mean_last']:>8.3f}{rel:>8.1f}%")
    print("=" * 78)

    with open('step5_uncertainty.json', 'w') as f:
        json.dump(dict(steps=STEPS, batch=BATCH, results=results), f, indent=2)
    print("[输出] 已存 step5_uncertainty.json")


if __name__ == '__main__':
    main()
