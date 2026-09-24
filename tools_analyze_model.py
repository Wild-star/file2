#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WAFT-Stereo 模型结构 / 参数量分布分析"""
import os, sys, json, math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.waft import WAFT
from bridgedepth.config import get_cfg

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "configs/SynLarge/DAv2S-4.yaml"

cfg = get_cfg()
cfg.merge_from_file(CONFIG)


def n_params(m, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    return sum(p.numel() for p in m.parameters())


def lora_params(m):
    """统计 LoRA 注入的参数"""
    tot = 0
    for n, p in m.named_parameters():
        if "lora_" in n:
            tot += p.numel()
    return tot


model = WAFT(cfg).eval()
total = n_params(model)
trainable = n_params(model, True)

print("=" * 78)
print(f" 配置: {CONFIG}")
print(f" 体系: FEATURE_ENCODER={cfg.WAFT.FEATURE_ENCODER.TYPE}/{cfg.WAFT.FEATURE_ENCODER.ARCH}"
      f"  LoRA r={cfg.WAFT.FEATURE_ENCODER.LORA_RANK} a={cfg.WAFT.FEATURE_ENCODER.LORA_ALPHA}")
print(f" 迭代: TASK={cfg.WAFT.ITERATIVE_MODULE.TASK}  → {len(cfg.WAFT.ITERATIVE_MODULE.TASK)} 次")
print("=" * 78)
print(f"\n【总参数量】{total/1e6:.3f} M")
print(f"  可训练   : {trainable/1e6:.3f} M  ({trainable/total*100:.1f}%)")
print(f"  冻结     : {(total-trainable)/1e6:.3f} M  ({(total-trainable)/total*100:.1f}%)")
print(f"  LoRA 参数: {lora_params(model)/1e6:.3f} M")

# ---- 顶层模块分解 ----
print("\n" + "=" * 78)
print("【顶层模块参数分布】")
print("=" * 78)
print(f"{'模块':<22}{'参数量':>14}{'占比':>10}{'可训练':>14}")
print("-" * 78)
rows = []
for name, child in model.named_children():
    p = n_params(child)
    t = n_params(child, True)
    rows.append((name, p, t))
for name, p, t in sorted(rows, key=lambda x: -x[1]):
    print(f"{name:<22}{p/1e6:>11.3f} M{p/total*100:>9.2f}%{t/1e6:>11.3f} M")
print("-" * 78)
print(f"{'合计':<22}{total/1e6:>11.3f} M{100.0:>9.2f}%{trainable/1e6:>11.3f} M")

# ---- encoder 内部 ----
print("\n" + "=" * 78)
print("【encoder (DAv2) 内部拆解】")
print("=" * 78)
enc = model.encoder
for name, child in enc.named_children():
    p = n_params(child)
    print(f"  {name:<24}{p/1e6:>11.3f} M  ({p/total*100:>5.2f}% of total)")
# DINOv2 backbone
enc_backbone = getattr(enc, "encoder", None)
if enc_backbone is not None:
    bp = n_params(enc_backbone)
    lp = lora_params(enc_backbone)
    print(f"  {'└ 主干(ViT+LoRA)':<24}{bp/1e6:>11.3f} M   其中 LoRA {lp/1e6:.3f} M, 冻结 {((bp-lp)/1e6):.3f} M")
    base = enc_backbone
    if hasattr(base, "base_model"):
        core = base.base_model.model
        try:
            nblk = len(core.pretrained.blocks)
            print(f"     主干层数: {nblk},  hidden dim: {core.pretrained.embed_dim if hasattr(core.pretrained,'embed_dim') else '?'}")
        except Exception:
            pass

# ---- 迭代模块内部 ----
for mod_name in ["prop_decoder", "delta_decoder"]:
    mod = getattr(model, mod_name, None)
    if mod is None:
        continue
    print("\n" + "=" * 78)
    print(f"【{mod_name} (VitIter/{getattr(mod,'dim','?')}) 内部拆解】")
    print("=" * 78)
    tot_mod = n_params(mod)
    for name, child in mod.named_children():
        p = n_params(child)
        print(f"  {name:<24}{p/1e6:>11.3f} M  ({p/tot_mod*100:>5.1f}% of module)")
    lp = lora_params(mod)
    print(f"  {'':<24}{'-'*13}")
    print(f"  {'模块合计':<24}{tot_mod/1e6:>11.3f} M   其中 LoRA {lp/1e6:.3f} M")

# ---- 头 ----
print("\n" + "=" * 78)
print("【预测头 & 投影层】")
print("=" * 78)
for name in ["prop_proj", "delta_proj", "prop_bins_head", "prop_mask_head",
             "delta_mask_head", "delta_dist_head", "delta_disp_head"]:
    m = getattr(model, name, None)
    if m is not None:
        print(f"  {name:<22}{n_params(m)/1e3:>10.2f} K")

# ---- 按功能归类 ----
print("\n" + "=" * 78)
print("【按功能归类】")
print("=" * 78)
groups = {
    "特征编码器(DAv2+DINOv2+LoRA)": n_params(model.encoder),
    "Prop 迭代模块": n_params(model.prop_decoder) + n_params(model.prop_proj),
    "Delta 迭代模块": n_params(model.delta_decoder) + n_params(model.delta_proj),
    "预测头(5个)": sum(n_params(getattr(model, n)) for n in
                    ["prop_bins_head", "prop_mask_head", "delta_mask_head",
                     "delta_dist_head", "delta_disp_head"]),
}
print(f"{'功能块':<32}{'参数量':>12}{'占比':>10}")
print("-" * 78)
for k, v in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"{k:<32}{v/1e6:>9.3f} M{v/total*100:>9.2f}%")
print("-" * 78)
print(f"{'合计':<32}{sum(groups.values())/1e6:>9.3f} M")

# ---- 前向 MACs ----
print("\n" + "=" * 78)
print("【前向 MACs（单次推理）】")
print("=" * 78)
try:
    H, W = 480, 640
    dummy = {"img1": torch.randn(1, 3, H, W), "img2": torch.randn(1, 3, H, W)}
    with torch.no_grad():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            model(dummy)
    macs = sum(int(e.flops) for e in prof.events()) / 2 / 1e9
    print(f"  输入 {H}x{W} → 前向 MACs ≈ {macs:.1f} G")
    with torch.no_grad():
        m2 = model(dummy)
    print(f"  输出: disp_pred {tuple(m2['disp_pred'].shape)}, "
          f"{len(m2['delta_disp_preds'])} 个 delta 预测 + init")
except Exception as e:
    print(f"  MACs 统计失败: {type(e).__name__}: {str(e)[:80]}")
