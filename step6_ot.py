#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6 — 光流领域模块融合：最优传输（OT）全局匹配初始化（FlowIt, 2026）

策略（按用户指示）：优先融合**非双目领域**的创新模块——跨域迁移本身即创新性说明。
本脚本把光流 FlowIt 的「Sinkhorn 最优传输全局匹配 + 置信度显式化」融合进 FusionWarp
的双目初始化：

  baseline  fusion    : 视差相关 + softmax 逐位置独立归一化（soft-argmax）
  融合后    fusion_ot : 视差相关 + Sinkhorn OT（行=左图位置、列=视差候选，
                        行列双约束 → 全局一致的 soft assignment），并显式输出
                        峰值置信度 conf 用于初始化监督加权

创新性说明（跨域迁移）：光流 FlowIt 的 OT 全局匹配 + 置信度引导被引入「相关 + warp
互补」的双目迭代框架，替代 softmax 逐位置归一化；OT 的列约束（uniqueness）抑制了
多个位置同时匹配到同一视差带来的歧义。

用法：
    python step6_ot.py            # STEPS=120 BATCH=4 对照
    STEPS=80 python step6_ot.py
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from step2_fusion_composite import (
    FeatureEncoder, MatchFeat, CrossAttentionLayer, GlobalMatcher, SparseCorrAnchor,
    GatedFusion, TokenSparseViT, FusionWarpStereo, disp_warp, convex_upsample,
    mixture_laplace_nll, make_synthetic_batch,
)

STEPS = int(os.environ.get('STEPS', '120'))
BATCH = int(os.environ.get('BATCH', '4'))
H, W, MAX_DISP, SEED, LR = 96, 128, 64, 0, 2e-3
D_CAND = 9           # 视差候选数（1/8 尺度，覆盖 0~32 全分辨率视差，与原版一致）


class OTGlobalMatcher(nn.Module):
    """FlowIt 式 OT 全局匹配：视差相关 + Sinkhorn OT + 峰值置信度。"""
    def __init__(self, C=32, D=24, heads=4, img_h=96, img_w=128, reg=0.5, iters=20):
        super().__init__()
        self.C, self.D, self.reg, self.iters = C, D, reg, iters
        self.h, self.w = img_h // 8, img_w // 8
        N = self.h * self.w
        self.pos = nn.Parameter(torch.randn(1, N, C) * 0.02)
        self.cross = CrossAttentionLayer(C, heads)
        self.proj = nn.Conv2d(2 * C, C, 1)

    def sinkhorn(self, cost):
        """cost: (B, N, D) 相似度 → OT 计划 P，行和=1、列和=N/D。"""
        B, N, D = cost.shape
        K = torch.exp(cost / self.reg)
        u = torch.ones(B, N, 1, device=cost.device)
        v = torch.ones(B, D, 1, device=cost.device)
        a = torch.ones(B, N, 1, device=cost.device)            # 行和 = 1
        b = torch.ones(B, D, 1, device=cost.device) * (N / D)  # 列和 = N/D
        for _ in range(self.iters):
            v = b / (torch.bmm(K.transpose(1, 2), u) + 1e-8)
            u = a / (torch.bmm(K, v) + 1e-8)
        return u * K * v.transpose(1, 2)                        # (B, N, D)

    def forward(self, f1, f2):
        B = f1.shape[0]
        f1c = F.avg_pool2d(f1, 4, 4)
        f2c = F.avg_pool2d(f2, 4, 4)
        l = f1c.flatten(2).transpose(1, 2) + self.pos
        r = f2c.flatten(2).transpose(1, 2) + self.pos
        l, r = self.cross(l, r)
        l = l.transpose(1, 2).reshape(B, self.C, self.h, self.w)
        r = r.transpose(1, 2).reshape(B, self.C, self.h, self.w)
        l = F.normalize(l, dim=1)
        cost = []
        for d in range(self.D):
            w = disp_warp(r, torch.full((B, 1, self.h, self.w), float(d), device=f1.device),
                          padding_mode='zeros')
            cost.append((l * F.normalize(w, dim=1)).sum(1))     # (B, h, w)
        cost = torch.stack(cost, dim=1)                          # (B, D, h, w)
        cost = cost.flatten(2).transpose(1, 2)                   # (B, N, D)
        P = self.sinkhorn(cost)                                  # (B, N, D)
        P = P.transpose(1, 2).reshape(B, self.D, self.h, self.w)
        idx = torch.arange(self.D, device=f1.device, dtype=f1.dtype).view(1, self.D, 1, 1)
        d_gm = (P * idx).sum(1, keepdim=True)                    # soft-argmax
        conf = P.max(1, keepdim=True)[0]                         # 峰值置信度
        d_gm = F.interpolate(d_gm, scale_factor=4, mode='bilinear', align_corners=True) * 4.0
        conf = F.interpolate(conf, scale_factor=4, mode='bilinear', align_corners=True)
        g_feat = F.interpolate(torch.cat([l, r], dim=1), scale_factor=4,
                               mode='bilinear', align_corners=True)
        g_feat = self.proj(g_feat)
        return d_gm, g_feat, conf


class OTFusionWarp(nn.Module):
    """FusionWarp + OTGlobalMatcher（其余与 FusionWarpStereo 相同）。"""
    def __init__(self, C=32, hidden=32, G=8, R=4, D=24, iters=2, img_h=96, img_w=128):
        super().__init__()
        self.encoder = FeatureEncoder(C)
        self.match = MatchFeat(C)
        self.global_matcher = OTGlobalMatcher(C, D, img_h=img_h, img_w=img_w)
        self.fusion = GatedFusion(C)
        self.delta_proj = nn.Conv2d(4 * C + 1, hidden, 3, padding=1)
        self.delta_decoder = TokenSparseViT(hidden, depth=2, heads=4, patch=8, out_ch=hidden, sparse=True)
        self.disp_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.mask_head = nn.Conv2d(hidden, 9 * 4, 3, padding=1)
        self.anchor = SparseCorrAnchor(C, G, R, hidden)
        self.iters = iters

    def forward(self, img1, img2):
        f1, f2 = self.encoder(img1), self.encoder(img2)
        m1, m2 = self.match(img1), self.match(img2)
        d_gm, g_feat, conf = self.global_matcher(f1, f2)
        disp = d_gm
        net = None
        preds, gates = [], []
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
            delta_disp = self.disp_head(net)
            mask = 0.25 * self.mask_head(net)
            disp = disp + delta_disp
            preds.append(convex_upsample(disp * 2, mask))
        return {'preds': preds, 'gates': gates, 'd_gm': d_gm, 'd_cv': None, 'conf': conf}


def train_one(name, model, use_conf):
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
            err_gm = (out['d_gm'] - gt_half).abs()
            if use_conf and out.get('conf') is not None:
                loss = loss + 0.5 * mixture_laplace_nll(err_gm) + \
                       0.5 * (err_gm * out['conf']).mean()      # 置信度引导初始化
            else:
                loss = loss + 0.5 * mixture_laplace_nll(err_gm)
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
                cmean = out['conf'].mean().item() if out.get('conf') is not None else float('nan')
            hist.append(dict(step=step, epe=epe, bad1=bad1, loss=loss.item(), conf_mean=cmean))

    dt = time.time() - t0
    res = dict(name=name, params=n_params, time_s=round(dt, 1),
               epe_first=hist[0]['epe'], epe_last=hist[-1]['epe'],
               bad1_last=hist[-1]['bad1'], conf_mean_last=hist[-1]['conf_mean'],
               steps=STEPS, hist=hist)
    print(f"  [{name:>14}] {STEPS}步 {dt:6.1f}s  EPE {hist[0]['epe']:6.3f} -> {hist[-1]['epe']:6.3f} px  "
          f"bad1px {hist[-1]['bad1']*100:5.1f}%  conf̄={hist[-1]['conf_mean']:.3f}", flush=True)
    return res


def main():
    configs = [
        ('fusion',      lambda: (FusionWarpStereo(img_h=H, img_w=W, iters=2, D_bins=D_CAND), False)),
        ('fusion_ot',   lambda: (OTFusionWarp(img_h=H, img_w=W, iters=2, D=D_CAND), True)),
    ]
    print(f"[对照] softmax 逐位置相关 vs Sinkhorn OT 全局匹配（同候选数 D={D_CAND}）\n")
    results = [train_one(n, m, u) for n, (m, u) in [(c[0], c[1]()) for c in configs]]

    base = results[0]['epe_last']
    print("\n" + "=" * 78)
    print("【对照】EPE 越小越好；相对 baseline (softmax) 的降幅")
    print("=" * 78)
    print(f"{'配置':<14}{'参数量':>9}{'EPE首':>9}{'EPE末':>9}{'bad1px':>8}{'conf̄':>8}{'相对降幅':>9}")
    print("-" * 78)
    for r in results:
        rel = (1 - r['epe_last'] / base) * 100
        print(f"{r['name']:<14}{r['params']/1e3:>7.1f}K{r['epe_first']:>9.3f}{r['epe_last']:>9.3f}"
              f"{r['bad1_last']*100:>7.1f}%{r['conf_mean_last']:>8.3f}{rel:>8.1f}%")
    print("=" * 78)

    with open('step6_ot.json', 'w') as f:
        json.dump(dict(steps=STEPS, batch=BATCH, results=results), f, indent=2)
    print("[输出] 已存 step6_ot.json")


if __name__ == '__main__':
    main()
