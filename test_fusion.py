#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FusionWarp 缝合自检脚本（在【有 torch 的环境】运行）。

用法：
    python test_fusion.py              # 模块级自检（快，不需要 encoder 权重）
    python test_fusion.py --full       # 完整 WAFT 前向+反向自检（需要 torch + timm/peft 等）

验证目标：
  1. 各融合模块前向形状正确；
  2. SparseCorrAnchor / GEVCostAnchor 的「零初始化」生效 → anchor 输出恒 0，
     保证 FUSION.ENABLED=True 且 USE_GLOBAL_INIT=False 时前向等价原版（手术安全）；
  3. GlobalMatcher 输出合理的初始视差；
  4. 完整 WAFT（DAv2 vits，小输入）前向 + 反向能跑通，梯度能流到匹配分支。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model.fusion import MatchingBranch, SparseCorrAnchor, GEVCostAnchor, GlobalMatcher, GatedFusion
from model.utils import disp_warp


def test_matching_branch():
    print("[1] MatchingBranch")
    mb = MatchingBranch(ch=32, out_ch=32)
    x = torch.randn(2, 3, 64, 96)
    y = mb(x)
    assert y.shape == (2, 32, 16, 24), f"期望 (2,32,16,24)，得到 {y.shape}"
    print(f"    ✅ 输入 {tuple(x.shape)} → 输出 {tuple(y.shape)}（1/4 分辨率）")


def test_sparse_corr_anchor():
    print("[2] SparseCorrAnchor（零初始化 = 手术安全）")
    C, G, R, out_ch = 32, 8, 4, 48
    anchor = SparseCorrAnchor(C=C, G=G, R=R, out_ch=out_ch)
    # 零初始化自检：权重/偏置全 0
    assert anchor.head.weight.abs().max().item() == 0.0
    assert anchor.head.bias.abs().max().item() == 0.0
    m1 = torch.randn(2, C, 16, 24)
    m2 = torch.randn(2, C, 16, 24)
    disp = torch.randn(2, 1, 16, 24) * 10
    out = anchor(m1, m2, disp)
    assert out.shape == (2, out_ch, 16, 24), f"输出形状 {out.shape}"
    assert out.abs().max().item() < 1e-6, f"零初始化输出应恒为 0，实际 max|out|={out.abs().max().item()}"
    print(f"    ✅ 输出形状 {tuple(out.shape)}，max|out|={out.abs().max().item():.2e}（恒 0）")


def test_gev_cost_anchor():
    print("[3] GEVCostAnchor（零初始化 = 手术安全）")
    C, out_ch = 32, 48
    for agg_kind in ['sep3d', 'full3d']:
        gev = GEVCostAnchor(C=C, G=4, K=9, R=8, Cv=8, out_ch=out_ch, agg_kind=agg_kind)
        m1 = torch.randn(2, C, 16, 24)
        m2 = torch.randn(2, C, 16, 24)
        disp = torch.randn(2, 1, 16, 24) * 10
        d_cv, feat = gev(m1, m2, disp)
        assert feat.shape == (2, out_ch, 16, 24), f"feat 形状 {feat.shape}"
        assert d_cv.shape == (2, 1, 16, 24), f"d_cv 形状 {d_cv.shape}"
        # 零初始化：feat=0，d_cv 退化为当前视差（对 softmax 均匀 → 加权平均 = 各候选中心 ≈ disp）
        assert feat.abs().max().item() < 1e-6, f"{agg_kind} feat 应恒 0"
        print(f"    ✅ [{agg_kind}] feat max|feat|={feat.abs().max().item():.2e}（恒 0）")


def test_global_matcher():
    print("[4] GlobalMatcher（可学习交叉注意力 + 全范围相关 → d_gm + g_feat）")
    C, heads, n_disp = 48, 4, 16
    gm = GlobalMatcher(C=C, heads=heads, n_disp=n_disp)
    f1 = torch.randn(2, C, 32, 40)   # 1/2 尺度
    f2 = torch.randn(2, C, 32, 40)
    d_gm, g_feat = gm(f1, f2)
    assert d_gm.shape == (2, 1, 32, 40), f"d_gm 形状 {d_gm.shape}"
    assert g_feat.shape == (2, C, 32, 40), f"g_feat 形状 {g_feat.shape}"
    assert (d_gm >= 0).all() and (d_gm <= (n_disp - 1) * 4).all(), \
        f"d_gm 值域应落在 [0, {(n_disp - 1) * 4}]，实际 [{d_gm.min().item():.2f}, {d_gm.max().item():.2f}]"
    # 反向：梯度能流到交叉注意力（含 QKV 投影）
    (d_gm.mean() + g_feat.mean()).backward()
    assert gm.cross.to_q.weight.grad is not None, "GlobalMatcher QKV 无梯度"
    assert gm.cross.ff[0].weight.grad is not None, "GlobalMatcher 无梯度"
    print(f"    ✅ d_gm {tuple(d_gm.shape)} 值域 [{d_gm.min().item():.2f}, {d_gm.max().item():.2f}]，"
          f"g_feat {tuple(g_feat.shape)}，QKV 梯度正常")


def test_gated_fusion():
    print("[5] GatedFusion（门控融合局部锚 + 全局上下文，零初始化）")
    C = 48
    gf = GatedFusion(C=C)
    # 零初始化自检：out_scale=0 → fused 恒 0
    assert gf.out_scale.item() == 0.0, "GatedFusion out_scale 应零初始化"
    anchor = torch.randn(2, C, 16, 24)
    g_feat = torch.randn(2, C, 16, 24)
    fused = gf(anchor, g_feat)
    assert fused.shape == (2, C, 16, 24), f"fused 形状 {fused.shape}"
    assert fused.abs().max().item() < 1e-6, f"零初始化 fused 应恒 0，实际 {fused.abs().max().item()}"
    fused.mean().backward()
    assert gf.gate.weight.grad is not None, "GatedFusion 无梯度"
    print(f"    ✅ fused 形状 {tuple(fused.shape)}，零初始化 max|fused|={fused.abs().max().item():.2e}（恒 0），梯度正常")


def test_waft_full():
    print("[6] 完整 WAFT 前向 + 反向（DAv2 vits，小输入）")
    from bridgedepth.config import get_cfg
    from algorithms.waft import WAFT

    cfg = get_cfg()
    cfg.merge_from_file("configs/SynLarge/DAv2S-4.yaml")
    cfg.WAFT.MAX_DISP = 128          # 缩小视差范围 → GlobalMatcher 16 个候选，加速自检
    cfg.WAFT.FUSION.ENABLED = True
    cfg.WAFT.FUSION.USE_ANCHOR = True
    cfg.WAFT.FUSION.USE_GLOBAL_INIT = False
    cfg.WAFT.FUSION.ANCHOR_KIND = "corr"
    cfg.freeze()

    model = WAFT(cfg)
    print(f"    模型可训练参数 {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

    sample = {
        "img1": torch.rand(1, 3, 128, 160) * 255,
        "img2": torch.rand(1, 3, 128, 160) * 255,
    }
    out = model(sample)
    assert 'disp_pred' in out and 'init' in out and 'delta_disp_preds' in out
    assert out['disp_pred'].shape == (1, 128, 160), f"disp_pred 形状 {out['disp_pred'].shape}"
    print(f"    ✅ 前向通过：disp_pred {tuple(out['disp_pred'].shape)}，init {tuple(out['init'].shape)}")

    # 反向：梯度能流到匹配分支（验证锚模块参与训练）
    loss = out['disp_pred'].abs().mean()
    loss.backward()
    assert model.matching_branch.net[0].weight.grad is not None, "匹配分支无梯度"
    print("    ✅ 反向通过：梯度已流到 matching_branch")

    # 打开 GlobalMatcher，验证前向仍跑通
    cfg2 = get_cfg()
    cfg2.merge_from_file("configs/SynLarge/DAv2S-4.yaml")
    cfg2.WAFT.MAX_DISP = 128
    cfg2.WAFT.FUSION.ENABLED = True
    cfg2.WAFT.FUSION.USE_ANCHOR = True
    cfg2.WAFT.FUSION.USE_GLOBAL_INIT = True
    cfg2.WAFT.FUSION.ANCHOR_KIND = "corr"
    cfg2.freeze()
    model2 = WAFT(cfg2)
    out2 = model2(sample)
    assert out2['disp_pred'].shape == (1, 128, 160)
    print("    ✅ USE_GLOBAL_INIT=True 前向通过（GlobalMatcher 替换初始视差）")

    # 打开 GatedFusion（门控融合局部锚 + 全局上下文）
    cfg3 = get_cfg()
    cfg3.merge_from_file("configs/SynLarge/DAv2S-4.yaml")
    cfg3.WAFT.MAX_DISP = 128
    cfg3.WAFT.FUSION.ENABLED = True
    cfg3.WAFT.FUSION.USE_ANCHOR = True
    cfg3.WAFT.FUSION.USE_GLOBAL_INIT = True
    cfg3.WAFT.FUSION.USE_GATED_FUSION = True
    cfg3.WAFT.FUSION.ANCHOR_KIND = "corr"
    cfg3.freeze()
    model3 = WAFT(cfg3)
    out3 = model3(sample)
    assert out3['disp_pred'].shape == (1, 128, 160)
    loss3 = out3['disp_pred'].abs().mean()
    loss3.backward()
    assert model3.gated_fusion.gate.weight.grad is not None, "GatedFusion 无梯度"
    print("    ✅ USE_GATED_FUSION=True 前向+反向通过（GatedFusion 门控融合生效）")


if __name__ == '__main__':
    test_matching_branch()
    test_sparse_corr_anchor()
    test_gev_cost_anchor()
    test_global_matcher()
    test_gated_fusion()
    print("\n[模块级自检全部通过]")
    if '--full' in sys.argv:
        test_waft_full()
        print("\n[完整 WAFT 自检通过]")
    else:
        print("\n提示：加 --full 参数可运行完整 WAFT 前向+反向自检（需 torch + timm/peft，且较慢）")
