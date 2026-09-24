#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 — WAVE 风格 baseline vs FusionWarp 的同条件对照（诚实定位）

背景：本设计核心 insight（相关体 + warp 互补）已被 WAVE-Stereo（arXiv:2607.13674）
抢先，故本工作定位为「WAVE-Stereo 的差异化变体 / 消融研究」。本脚本在自包含
POC（合成数据、CPU）层面回答：

  在「相关 + warp 互补」这一共同框架下，本设计相对 WAVE 风格的三个实现选择
  （① GlobalMatcher 全局匹配初始化、② TokenSparseViT 稀疏解码、③ GatedFusion 门控融合）
  是否带来增益？

对照配置（同数据/种子/步数）：
  wave               : WAVE 风格（相关 2D 聚合初始化 + ConvGRU + concat 融合）
  fusion             : 本设计（GlobalMatcher + TokenSparseViT + 门控融合 + 窄带相关锚）
  fusion_noanchor    : fusion 去掉相关锚（use_anchor=False）      —— 测「锚」贡献
  fusion_nosparse    : fusion 关闭 token 稀疏（use_sparse=False）  —— 测「稀疏」贡献

注意：这是 CPU 合成数据的 POC 信号，仅用于自检与方向判断；最终结论须在
WAVE-Stereo 的 9 数据集全量训练设定下复验（见 docs/COMPARISON_PLAN.md）。

用法：
    python step4_comparison.py            # 默认 STEPS=120 BATCH=4
    STEPS=60 python step4_comparison.py
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
    TokenSparseViT, FusionWarpStereo, disp_warp, convex_upsample, group_corr,
    mixture_laplace_nll, make_synthetic_batch,
)

STEPS = int(os.environ.get('STEPS', '120'))
BATCH = int(os.environ.get('BATCH', '4'))
H, W, MAX_DISP, SEED, LR = 96, 128, 64, 0, 2e-3


# --------------------------------------------------------------------------- #
# WAVE 风格组件（自包含镜像，仅用于对照；非 WAVE-Stereo 官方实现）
# --------------------------------------------------------------------------- #
class ConvGRU(nn.Module):
    """极简卷积 GRU（对应 WAVE-Stereo 的 ConvGRU 迭代单元）。"""
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.hidden = hidden
        self.ih = nn.Conv2d(in_ch, 3 * hidden, 3, padding=1)
        self.hh = nn.Conv2d(hidden, 3 * hidden, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden, x.shape[2], x.shape[3], device=x.device)
        gi = self.ih(x)
        gh = self.hh(h)
        zi, ri, ni = torch.chunk(gi, 3, dim=1)
        zh, rh, nh = torch.chunk(gh, 3, dim=1)
        z = torch.sigmoid(zi + zh)
        r = torch.sigmoid(ri + rh)
        n = torch.tanh(ni + r * nh)
        return (1 - z) * n + z * h


class Corr2DInit(nn.Module):
    """WAVE 式初始视差：全范围（D 候选）group-wise 相关 + 2D 卷积聚合 + soft-argmin。"""
    def __init__(self, C=32, G=8, D=24, Cv=16):
        super().__init__()
        self.G, self.D = G, D
        self.agg = nn.Sequential(
            nn.Conv2d(G * D, Cv * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(Cv * 2, Cv, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Conv2d(Cv, D, 1)

    def forward(self, m1, m2):
        B, C, h, w = m1.shape
        corrs = []
        for d in range(self.D):
            m2s = disp_warp(m2, torch.full((B, 1, h, w), float(d), device=m1.device))
            corrs.append(group_corr(m1, m2s, self.G))          # (B, G, h, w)
        vol = torch.stack(corrs, dim=1)                        # (B, D, G, h, w)
        vol = vol.reshape(B, self.D * self.G, h, w)
        prob = self.head(self.agg(vol))                        # (B, D, h, w)
        prob = F.softmax(prob, dim=1)
        idx = torch.arange(self.D, device=m1.device, dtype=m1.dtype).view(1, self.D, 1, 1)
        return (prob * idx).sum(dim=1, keepdim=True)           # (B, 1, h, w)


class WaveStyleBaseline(nn.Module):
    """WAVE 风格自包含版：相关 2D 聚合初始化 + GWCE 式三分支 + ConvGRU + concat 融合。

    与 WAVE-Stereo 的对应：GWCE 的三分支 = corr(相关检索) + disp 先验 + warp 对齐；
    融合用 concat（无门控）；无全局匹配初始化（用 Corr2DInit）、无 token 稀疏。
    """
    def __init__(self, C=32, hidden=32, G=8, R=4, D=24, iters=2):
        super().__init__()
        self.encoder = FeatureEncoder(C)
        self.init = Corr2DInit(C, G, D)
        self.enc_c = nn.Sequential(nn.Conv2d((2 * R + 1) * G, hidden, 1),
                                   nn.Conv2d(hidden, hidden, 3, padding=1))
        self.enc_d = nn.Sequential(nn.Conv2d(1, hidden, 7, padding=3), nn.ReLU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1))
        self.enc_w = nn.Sequential(nn.Conv2d(2 * C, hidden, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1))
        self.fusion = nn.Conv2d(3 * hidden, hidden, 3, padding=1)
        self.gru = ConvGRU(hidden + 1, hidden)
        self.disp_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.mask_head = nn.Conv2d(hidden, 9 * 4, 3, padding=1)
        self.R, self.G, self.iters = R, G, iters

    def _corr_lookup(self, m1, m2, disp):
        B, C, h, w = m1.shape
        offs = torch.arange(-self.R, self.R + 1, device=m1.device, dtype=m1.dtype)
        corrs = []
        for o in offs:
            m2s = disp_warp(m2, disp + o, padding_mode='zeros')
            corrs.append(group_corr(m1, m2s, self.G))          # (B, G, h, w)
        vol = torch.stack(corrs, dim=1)                        # (B, 2R+1, G, h, w)
        return vol.reshape(B, (2 * self.R + 1) * self.G, h, w)

    def forward(self, img1, img2):
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)                                # 1/2
        disp = self.init(f1, f2)                               # 初始视差（用于监督）
        d_gm = disp.detach() * 1.0
        h = None
        preds = []
        for _ in range(self.iters):
            disp = disp.detach()
            x_c = self.enc_c(self._corr_lookup(f1, f2, disp))
            x_d = self.enc_d(disp)
            warped = disp_warp(f2, disp, padding_mode='zeros')
            x_w = self.enc_w(torch.cat([f1, warped], dim=1))
            fused = self.fusion(torch.cat([x_c, x_d, x_w], dim=1))
            m = torch.cat([fused, disp], dim=1)
            h = self.gru(m, h)
            delta = self.disp_head(h)
            mask = 0.25 * self.mask_head(h)
            disp = disp + delta
            preds.append(convex_upsample(disp * 2, mask))
        return {'preds': preds, 'gates': [], 'd_gm': d_gm, 'd_cv': None}


# --------------------------------------------------------------------------- #
# 训练 + 对照
# --------------------------------------------------------------------------- #
def loss_of(out, gt, gt_half, valid):
    loss = 0.0
    if out.get('d_gm') is not None:
        loss = loss + 0.5 * mixture_laplace_nll((out['d_gm'] - gt_half).abs())
    if out.get('d_cv') is not None:
        loss = loss + 0.3 * mixture_laplace_nll((out['d_cv'] - gt_half).abs())
    for i, p in enumerate(out['preds']):
        w = 0.5 ** (len(out['preds']) - 1 - i)
        loss = loss + w * mixture_laplace_nll((p - gt).abs()[valid])
    if out.get('gates'):
        loss = loss + 0.02 * torch.stack(out['gates']).mean()
    return loss


def train_one(name, build_fn):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_fn().train()
    n_params = sum(p.numel() for p in model.parameters())

    left, right, gt, valid = make_synthetic_batch(BATCH, H, W, MAX_DISP, seed=SEED)
    gt_half = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)

    hist = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        out = model(left, right)
        loss = loss_of(out, gt, gt_half, valid)
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
    print(f"  [{name:>16}] {STEPS}步 {dt:6.1f}s  EPE {hist[0]['epe']:6.3f} -> {hist[-1]['epe']:6.3f} px  "
          f"bad1px {hist[-1]['bad1']*100:5.1f}%", flush=True)
    return res


def main():
    configs = [
        ('wave',            lambda: WaveStyleBaseline(iters=2)),
        ('fusion',          lambda: FusionWarpStereo(img_h=H, img_w=W, iters=2)),
        ('fusion_noanchor', lambda: FusionWarpStereo(img_h=H, img_w=W, iters=2, use_anchor=False)),
        ('fusion_nosparse', lambda: FusionWarpStereo(img_h=H, img_w=W, iters=2, use_sparse=False)),
    ]
    print(f"[对照] {len(configs)} 配置 × {STEPS} 步 × batch {BATCH}（同数据/种子）\n")
    results = [train_one(name, fn) for name, fn in configs]

    base = results[0]['epe_last']           # wave baseline
    print("\n" + "=" * 86)
    print("【对照表】EPE 越小越好；相对 wave baseline 的降幅（正=优于 baseline）")
    print("=" * 86)
    print(f"{'配置':<16}{'参数量':>9}{'EPE首':>9}{'EPE末':>9}{'bad1px':>8}{'耗时s':>8}{'相对wave':>9}")
    print("-" * 86)
    for r in results:
        rel = (1 - r['epe_last'] / base) * 100
        print(f"{r['name']:<16}{r['params']/1e3:>7.1f}K{r['epe_first']:>9.3f}{r['epe_last']:>9.3f}"
              f"{r['bad1_last']*100:>7.1f}%{r['time_s']:>8.1f}{rel:>8.1f}%")
    print("=" * 86)

    with open('step4_comparison.json', 'w') as f:
        json.dump(dict(steps=STEPS, batch=BATCH, results=results), f, indent=2)
    print("[输出] 已存 step4_comparison.json")


if __name__ == '__main__':
    main()
