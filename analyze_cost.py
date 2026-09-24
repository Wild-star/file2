#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FusionWarp 计算量(FLOPs) + 显存峰值 分析。

在训练尺寸下逐模块测量，并对比 WAFT 整体（基线 vs 融合）的前向 FLOPs 与显存。
用法：python analyze_cost.py
"""
import torch
from torch.utils.flop_counter import FlopCounterMode

from model.fusion import MatchingBranch, SparseCorrAnchor, GEVCostAnchor, GlobalMatcher, GatedFusion

device = 'cuda' if torch.cuda.is_available() else 'cpu'
B = 1
H, W = 480, 640                      # DAv2S-4 训练 crop 尺寸
fmap_H, fmap_W = H // 2, W // 2       # 1/2 尺度（fmap 分辨率）
m_H, m_W = H // 4, W // 4             # 1/4 尺度（匹配特征分辨率）


def flops_of(fn):
    fc = FlopCounterMode(display=False)
    with fc:
        fn()
    return fc.get_total_flops()


def mem_of(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    fn()
    return torch.cuda.max_memory_allocated()


def fmt_flops(f):
    return f"{f / 1e9:.3f} GFLOPs"


def fmt_mem(m):
    return f"{m / 1024**2:.1f} MB"


print(f"[设备] {device}  |  训练尺寸 {H}x{W}  batch={B}")
print("=" * 90)

# 1. MatchingBranch
mb = MatchingBranch(ch=32, out_ch=32).to(device).eval()
x = torch.randn(B, 3, H, W, device=device)
f = flops_of(lambda: mb(x)); m = mem_of(lambda: mb(x))
print(f"[1] MatchingBranch      输入 {H}x{W} → {m_H}x{m_W}  |  {fmt_flops(f)}  |  显存 {fmt_mem(m)}")

# 2. SparseCorrAnchor（R=4，9 次 warp + group corr）
corr = SparseCorrAnchor(C=32, G=8, R=4, out_ch=48).to(device).eval()
m1 = torch.randn(B, 32, m_H, m_W, device=device)
m2 = torch.randn(B, 32, m_H, m_W, device=device)
disp = torch.randn(B, 1, m_H, m_W, device=device)
f = flops_of(lambda: corr(m1, m2, disp)); m = mem_of(lambda: corr(m1, m2, disp))
print(f"[2] SparseCorrAnchor    相关±R=4  |  {fmt_flops(f)}  |  显存 {fmt_mem(m)}")

# 3. GEVCostAnchor（K=9，3D 聚合）
gev = GEVCostAnchor(C=32, G=4, K=9, R=8, Cv=8, out_ch=48, agg_kind='sep3d').to(device).eval()
f = flops_of(lambda: gev(m1, m2, disp)); m = mem_of(lambda: gev(m1, m2, disp))
print(f"[3] GEVCostAnchor(sep3d) K=9 3D聚合  |  {fmt_flops(f)}  |  显存 {fmt_mem(m)}")

# 4. GlobalMatcher（n_disp=100 全范围相关 + QKV 交叉注意力）
gm = GlobalMatcher(C=48, heads=4, n_disp=100).to(device).eval()
f1 = torch.randn(B, 48, fmap_H, fmap_W, device=device)
f2 = torch.randn(B, 48, fmap_H, fmap_W, device=device)
f = flops_of(lambda: gm(f1, f2)); m = mem_of(lambda: gm(f1, f2))
print(f"[4] GlobalMatcher        n_disp=100 + QKV  |  {fmt_flops(f)}  |  显存 {fmt_mem(m)}")

# 5. GatedFusion
gf = GatedFusion(C=48).to(device).eval()
anchor = torch.randn(B, 48, fmap_H, fmap_W, device=device)
g_feat = torch.randn(B, 48, fmap_H, fmap_W, device=device)
f = flops_of(lambda: gf(anchor, g_feat)); m = mem_of(lambda: gf(anchor, g_feat))
print(f"[5] GatedFusion         门控融合  |  {fmt_flops(f)}  |  显存 {fmt_mem(m)}")

print("=" * 90)

# 6. WAFT 整体对比（基线 vs 融合）
from bridgedepth.config import get_cfg
from algorithms.waft import WAFT


def build_waft(use_fusion, use_gated=False, use_global=False):
    cfg = get_cfg()
    cfg.merge_from_file("configs/SynLarge/DAv2S-4.yaml")
    cfg.WAFT.FUSION.ENABLED = use_fusion
    cfg.WAFT.FUSION.USE_ANCHOR = use_fusion
    cfg.WAFT.FUSION.USE_GLOBAL_INIT = use_global
    cfg.WAFT.FUSION.USE_GATED_FUSION = use_gated
    cfg.WAFT.FUSION.ANCHOR_KIND = "corr"
    cfg.freeze()
    return WAFT(cfg).to(device).eval()


sample = {"img1": torch.randn(B, 3, H, W, device=device) * 255,
          "img2": torch.randn(B, 3, H, W, device=device) * 255}

print("\n[整体] WAFT 前向 FLOPs + 显存（含 DAv2 编码器）")
print("-" * 90)
for name, m in [("基线(无 fusion)", build_waft(False)),
                ("+corr 锚", build_waft(True)),
                ("+corr+GlobalMatcher", build_waft(True, use_global=True)),
                ("完整融合(corr+GM+Gated)", build_waft(True, use_gated=True, use_global=True))]:
    with torch.no_grad():
        f = flops_of(lambda: m(sample))
        m_mem = mem_of(lambda: m(sample))
    print(f"  {name:<28}  {fmt_flops(f):>14}  |  显存 {fmt_mem(m_mem)}")
