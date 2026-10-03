#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 10 — 回归原始 WAFT baseline（纯 warp-only，无代价体）

决策依据（docs/WAFT_IMPROVEMENTS.md）：此前在 WAFT 上一次性堆叠 GlobalMatcher/锚/门控/
token 稀疏/OT/σ 等 7+ 个模块，多种子证明均无可稳健主张的增益。故**回归原始 WAFT**，
本脚本是自包含的纯 warp-only 版：prop bins 分类初始化 + warp 残差 + ViT 迭代，
不含任何代价体/相关锚/全局匹配/门控。作为「代价体 + WAFT 最小注入」的干净起点。

用法：
    python step10_waft_baseline.py          # 多种子报告 baseline EPE
    STEPS=80 SEEDS=0,1,2 python step10_waft_baseline.py
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from step2_fusion_composite import FeatureEncoder, TokenSparseViT, disp_warp, convex_upsample
from step7_difficulty import make_difficulty_batch

STEPS = int(os.environ.get('STEPS', '80'))
BATCH = int(os.environ.get('BATCH', '4'))
H, W, MAX_DISP, LR = 96, 128, 64, 2e-3
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
N_BINS = 9


class WAFTBaseline(nn.Module):
    """自包含纯 warp-only WAFT：prop bins 初始化 + delta 迭代（ViT），无代价体。"""
    def __init__(self, C=32, hidden=32, n_bins=9, iters=2):
        super().__init__()
        self.encoder = FeatureEncoder(C)
        self.prop_proj = nn.Conv2d(2 * C, hidden, 3, padding=1)
        self.prop_decoder = TokenSparseViT(hidden, depth=2, heads=4, patch=8, out_ch=hidden, sparse=False)
        self.prop_bins_head = nn.Conv2d(hidden, n_bins, 3, padding=1)
        self.delta_proj = nn.Conv2d(2 * C + hidden + 1, hidden, 3, padding=1)
        self.delta_decoder = TokenSparseViT(hidden, depth=2, heads=4, patch=8, out_ch=hidden, sparse=False)
        self.delta_disp_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.delta_mask_head = nn.Conv2d(hidden, 9 * 4, 3, padding=1)
        self.n_bins = n_bins
        self.iters = iters

    def forward(self, img1, img2):
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)                                   # 1/2
        idx_2x = torch.linspace(0, MAX_DISP / 2, self.n_bins, device=f1.device,
                                dtype=f1.dtype).view(1, self.n_bins, 1, 1)
        ph = self.prop_proj(torch.cat([f1, f2], dim=1))
        ph = self.prop_decoder(ph)[0]
        prob = F.softmax(self.prop_bins_head(ph), dim=1)
        disp = (prob * idx_2x).sum(1, keepdim=True)               # 1/2 尺度初始视差
        net = None
        preds = []
        for _ in range(self.iters):
            disp = disp.detach()
            warped = disp_warp(f2, disp, padding_mode='zeros')
            if net is None:
                net = torch.zeros_like(f1)
            net = self.delta_proj(torch.cat([f1, warped, net, disp], dim=1))
            net = self.delta_decoder(net)[0]
            delta = self.delta_disp_head(net)
            mask = 0.25 * self.delta_mask_head(net)
            disp = disp + delta
            preds.append(convex_upsample(disp * 2, mask))
        return {'preds': preds, 'd_gm': None, 'd_cv': None, 'gates': []}


def train_one(data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WAFTBaseline(iters=2).train()
    left, right, gt, valid = data

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)

    for step in range(1, STEPS + 1):
        out = model(left, right)
        loss = 0.0
        for i, p in enumerate(out['preds']):
            w = 0.5 ** (len(out['preds']) - 1 - i)
            loss = loss + w * (p - gt).abs()[valid].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        epe = (model(left, right)['preds'][-1] - gt).abs()[valid].mean().item()
    return epe, sum(p.numel() for p in model.parameters())


def main():
    easy = make_difficulty_batch(BATCH, H, W, MAX_DISP, seed=0, texture='smooth', occlusion=False)
    print(f"[WAFT baseline] 纯 warp-only，prop bins 初始化，无代价体  |  {SEEDS} 种子\n")
    epes, nparams = [], 0
    for s in SEEDS:
        e, n = train_one(easy, s)
        epes.append(e)
        nparams = n
        print(f"  [seed{s}] EPE {e:.3f}", flush=True)
    arr = np.array(epes)
    print(f"\n[结果] WAFT baseline EPE = {arr.mean():.3f} ± {arr.std():.3f}  ({nparams/1e3:.1f}K 参数)")
    json.dump(dict(steps=STEPS, seeds=SEEDS, epes=epes, mean=arr.mean(), std=arr.std(),
                   n_params=nparams), open('step10_waft_baseline.json', 'w'), indent=2)
    print("[输出] 已存 step10_waft_baseline.json")


if __name__ == '__main__':
    main()
