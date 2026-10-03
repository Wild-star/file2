#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 11 — 候选 A：代价体最小 concat 注入（WAFT + cost_feat）

候选 A（docs/survey_stereo_flow.md §6.3）：在原始 WAFT 上**只加一个窄带相关特征 cost_feat**
到 delta_proj 输入，其余（prop bins 初始化 / ViT 迭代 / warp / L1 损失）全部保持原样。

  原版: net = delta_proj(cat[fmap1, warped_fmap2, net, disp])
  候选A: net = delta_proj(cat[fmap1, warped_fmap2, cost_feat, net, disp])
         cost_feat = 1x1conv(窄带 group-wise 相关 vol(disp±R))   # 匹配证据特征

对照：WAFT（use_cost=False）vs WAFT+Cost（use_cost=True），3 种子 × 固定数据，L1 损失。
判定：WAFT+Cost 的 mean 稳健优于 WAFT（且 std 不重叠）→ 代价体注入稳健有效。

用法：python step11_cost_injection.py   （STEPS=80 SEEDS=0,1,2）
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from step2_fusion_composite import FeatureEncoder, TokenSparseViT, disp_warp, convex_upsample, group_corr
from step7_difficulty import make_difficulty_batch

STEPS = int(os.environ.get('STEPS', '80'))
BATCH = int(os.environ.get('BATCH', '4'))
H, W, MAX_DISP, LR = 96, 128, 64, 2e-3
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
N_BINS, G, R, COST_CH = 9, 8, 4, 16


class WAFTPlusCost(nn.Module):
    """原始 WAFT（warp-only）+ 可选 cost_feat 注入（use_cost 开关，单因子）。"""
    def __init__(self, C=32, hidden=32, n_bins=9, iters=2, use_cost=True):
        super().__init__()
        self.encoder = FeatureEncoder(C)
        self.prop_proj = nn.Conv2d(2 * C, hidden, 3, padding=1)
        self.prop_decoder = TokenSparseViT(hidden, depth=2, heads=4, patch=8, out_ch=hidden, sparse=False)
        self.prop_bins_head = nn.Conv2d(hidden, n_bins, 3, padding=1)

        self.use_cost = use_cost
        in_ch = 2 * C + hidden + 1
        if use_cost:
            self.cost_proj = nn.Conv2d((2 * R + 1) * G, COST_CH, 1)   # 窄带相关体 → cost_feat
            in_ch += COST_CH
        self.delta_proj = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.delta_decoder = TokenSparseViT(hidden, depth=2, heads=4, patch=8, out_ch=hidden, sparse=False)
        self.delta_disp_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.delta_mask_head = nn.Conv2d(hidden, 9 * 4, 3, padding=1)
        self.n_bins = n_bins
        self.iters = iters

    def corr_feature(self, m1, m2, disp):
        """窄带 group-wise 相关体（disp±R）→ cost_feat（1x1 投影）。"""
        B, C, h, w = m1.shape
        offs = torch.arange(-R, R + 1, device=m1.device, dtype=m1.dtype)
        corrs = []
        for o in offs:
            m2s = disp_warp(m2, disp + o, padding_mode='zeros')
            corrs.append(group_corr(m1, m2s, G))                    # (B, G, h, w)
        vol = torch.stack(corrs, dim=1).reshape(B, (2 * R + 1) * G, h, w)
        return self.cost_proj(vol)                                   # (B, COST_CH, h, w)

    def forward(self, img1, img2):
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)                                      # 1/2
        idx_2x = torch.linspace(0, MAX_DISP / 2, self.n_bins, device=f1.device,
                                dtype=f1.dtype).view(1, self.n_bins, 1, 1)
        ph = self.prop_proj(torch.cat([f1, f2], dim=1))
        ph = self.prop_decoder(ph)[0]
        prob = F.softmax(self.prop_bins_head(ph), dim=1)
        disp = (prob * idx_2x).sum(1, keepdim=True)
        net = None
        preds = []
        for _ in range(self.iters):
            disp = disp.detach()
            warped = disp_warp(f2, disp, padding_mode='zeros')
            if net is None:
                net = torch.zeros_like(f1)
            if self.use_cost:
                x = torch.cat([f1, warped, self.corr_feature(f1, f2, disp), net, disp], dim=1)
            else:
                x = torch.cat([f1, warped, net, disp], dim=1)
            net = self.delta_proj(x)
            net = self.delta_decoder(net)[0]
            delta = self.delta_disp_head(net)
            mask = 0.25 * self.delta_mask_head(net)
            disp = disp + delta
            preds.append(convex_upsample(disp * 2, mask))
        return {'preds': preds}


def train_one(use_cost, data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WAFTPlusCost(iters=2, use_cost=use_cost).train()
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
    print(f"[候选A] WAFT vs WAFT+Cost（窄带相关 cost_feat concat 注入）| {SEEDS} 种子 × 80 步\n")

    results = {}
    for tag, use_cost in [('WAFT', False), ('WAFT+Cost', True)]:
        epes, nparams = [], 0
        for s in SEEDS:
            e, n = train_one(use_cost, easy, s)
            epes.append(e)
            nparams = n
            print(f"  [{tag} seed{s}] EPE {e:.3f}", flush=True)
        arr = np.array(epes)
        results[tag] = dict(mean=arr.mean(), std=arr.std(), seeds=epes, n_params=nparams)

    base = results['WAFT']['mean']
    print("\n" + "=" * 62)
    print("【候选A 对照】mean±std；Δ = 相对 WAFT 的 EPE 降幅")
    print("=" * 62)
    for tag in ['WAFT', 'WAFT+Cost']:
        r = results[tag]
        rel = (1 - r['mean'] / base) * 100
        print(f"{tag:<12} {r['mean']:>8.3f}±{r['std']:.3f}  ({r['n_params']/1e3:.1f}K)  {rel:>+7.1f}%")
    print("=" * 62)
    d = results['WAFT+Cost']['mean'] - base
    print(f"ΔEPE = {d:+.3f}  |  {'✅ 代价体注入稳健有效' if d < -results['WAFT']['std'] else '❌ 不稳健/无增益'}")

    json.dump(dict(steps=STEPS, seeds=SEEDS, results=results), open('step11_cost_injection.json', 'w'), indent=2)
    print("[输出] 已存 step11_cost_injection.json")


if __name__ == '__main__':
    main()
