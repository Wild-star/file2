#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0① WAFT-Stereo 迭代的 token 级稀疏性统计

问题：delta 模块每轮在 2040 个 token 上跑 12 层 ViT（占单轮 49% 时间）。
      如果大部分 token 的更新量极小，就可以做 token 稀疏迭代。

测什么：
  1. 每轮迭代中，每个 token 对应的视差更新幅度 (Δdisp)
  2. 按幅度排序后，top-k% token 覆盖了多少更新总量  ← 稀疏度上限
  3. 活跃 token 占比 + 相邻轮次的活跃集合重叠率（对齐 PIP 的 hit ratio 分析）
  4. ViT 内部 token 特征变化量（次要指标）
输出：统计表 + 稀疏度曲线数据(json/csv)
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from bridgedepth.config import get_cfg
from algorithms.waft import WAFT

CFG = "configs/SynLarge/DAv2S-4.yaml"
CKPT = "ckpts/SynLarge/DAv2S-4.pth"
DATA = "datasets/ETH3D/two_view_training"
N_SCENES = int(os.environ.get("N_SCENES", "27"))
TARGET_H = 480                      # 统一缩放到高 480，宽按比例对齐到 16 的倍数

cfg = get_cfg(); cfg.merge_from_file(CFG)
device = "cuda"
model = WAFT(cfg).eval().to(device)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
# ckpt 结构: {'model': state_dict, ...}  —— 必须取 'model'，否则只加载到 5 个顶层键
sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
sd = {k.replace("module.", ""): v for k, v in sd.items()}
missing = model.load_state_dict(sd, strict=False)
loaded = len(sd) - len(missing.unexpected_keys)
print(f"[模型] ckpt 权重键 {len(sd)} 个 | 载入 {loaded} | missing={len(missing.missing_keys)}", flush=True)
assert loaded > 100, f"权重几乎没加载上，检查 ckpt 结构！loaded={loaded}"

N_ITER = len(cfg.WAFT.ITERATIVE_MODULE.TASK)

# ---- 钩子：抓 ViT 输入 token / 输出 token ----
vit_in, vit_out = [], []
h1 = model.delta_decoder.patch_embed.register_forward_hook(lambda m, i, o: vit_in.append(o.detach()))
h2 = model.delta_decoder.blks[-1].register_forward_hook(lambda m, i, o: vit_out.append(o.detach()))


def load_scene(d):
    im0 = np.asarray(Image.open(os.path.join(d, "im0.png")).convert("RGB"))
    im1 = np.asarray(Image.open(os.path.join(d, "im1.png")).convert("RGB"))
    t0 = torch.from_numpy(im0).permute(2, 0, 1).float()
    t1 = torch.from_numpy(im1).permute(2, 0, 1).float()
    H, W = t0.shape[-2:]
    h = TARGET_H
    w = int(round(W * h / H / 16)) * 16
    t0 = F.interpolate(t0[None], (h, w), mode="bilinear", align_corners=True)[0]
    t1 = F.interpolate(t1[None], (h, w), mode="bilinear", align_corners=True)[0]
    return t0[None].to(device), t1[None].to(device)


scenes = sorted(x for x in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, x)))[:N_SCENES]
print(f"[数据] {len(scenes)} 个 ETH3D 场景，目标高 {TARGET_H}", flush=True)

# 累积容器
delta_disp_iters = [[] for _ in range(N_ITER)]       # 每轮: 每 token 的 Δdisp  (flatten over scenes)
token_feat_chg = [[] for _ in range(N_ITER)]          # 每轮: ViT token 特征变化量
abs_disp = []
ntok = None

for si, sc in enumerate(scenes):
    x1, x2 = load_scene(os.path.join(DATA, sc))
    vit_in.clear(); vit_out.clear()
    with torch.no_grad():
        out = model({"img1": x1, "img2": x2})
    dps = out["delta_disp_preds"]                      # list of (B,1,H',W')  ← 每轮输出的 disp
    prev = None
    for t in range(N_ITER):
        cur = dps[t]
        if prev is not None:
            d = (cur - prev).abs()[0, 0]               # (H',W')  像素级 Δdisp
            Hp, Wp = d.shape
            gh, gw = Hp // 8, Wp // 8                  # token 网格 = 1/16 原图
            dh, dw = Hp // gh, Wp // gw
            tok = d[: gh * dh, : gw * dw].reshape(gh, dh, gw, dw).mean(dim=(1, 3))
            delta_disp_iters[t].append(tok.flatten().cpu().numpy())
            ntok = tok.numel()
        else:
            abs_disp.append(dps[0].abs().mean().item())
        prev = cur
    # ViT token 特征变化
    for t in range(len(vit_in)):
        vi, vo = vit_in[t], vit_out[t]
        vi = vi.flatten(2); vo = vo.flatten(2)
        if t > 0 and vit_in[t - 1].shape == vi.shape:
            chg = (vo - vit_in[t - 1].flatten(2)).norm(dim=1)[0]      # 近似：输出 vs 上一轮输入
        else:
            chg = vo.norm(dim=1)[0]
        token_feat_chg[min(t, N_ITER - 1)].append(chg.cpu().numpy())
    if (si + 1) % 5 == 0:
        print(f"  ... {si+1}/{len(scenes)}", flush=True)

h1.remove(); h2.remove()

# ---------------- 统计 ----------------
print("\n" + "=" * 84)
print("【P0① Token 稀疏性统计】  ETH3D, {} 场景, token 网格 {} 个".format(len(scenes), ntok))
print("=" * 84)

print(f"\n{'轮次':<6}{'Δdisp 中位数':>14}{'Δdisp 均值':>13}{'>1%阈值 活跃占比':>18}{'top-10% 覆盖量':>17}{'top-20% 覆盖量':>17}")
print("-" * 84)
report = {}
curve_data = {}
for t in range(1, N_ITER):
    arr = np.concatenate(delta_disp_iters[t])
    absarr = np.abs(arr)
    thr = 0.01 * absarr.max() if absarr.max() > 0 else 0
    active = float((absarr > thr).mean())
    # 覆盖率曲线：按幅度降序，top-k% 占总量比例
    s = np.sort(absarr)[::-1]
    cum = np.cumsum(s) / s.sum()
    def cover(p):
        k = max(1, int(len(s) * p / 100))
        return float(cum[k - 1])
    report[t] = dict(median=float(np.median(absarr)), mean=float(absarr.mean()),
                     active_ratio=active, top10=cover(10), top20=cover(20), top30=cover(30), top50=cover(50))
    curve_data[t] = [cover(p) for p in range(1, 101)]
    print(f"{t+1:<6}{np.median(absarr):>14.4f}{absarr.mean():>13.4f}{active*100:>17.1f}%{cover(10)*100:>16.1f}%{cover(20)*100:>16.1f}%")

print("-" * 84)

# 相邻轮次活跃集合重叠率（PIP 的 hit ratio）
print("\n【相邻轮次的活跃 token 重叠率】(Jaccard，越高说明越冗余)")
for t in range(2, N_ITER):
    a = np.abs(np.concatenate(delta_disp_iters[t - 1]))
    b = np.abs(np.concatenate(delta_disp_iters[t]))
    n = min(len(a), len(b))
    ta, tb = np.quantile(a[:n], 0.9), np.quantile(b[:n], 0.9)
    sa, sb = set(np.where(a[:n] > ta)[0].tolist()), set(np.where(b[:n] > tb)[0].tolist())
    jac = len(sa & sb) / max(1, len(sa | sb))
    print(f"  第 {t} 轮 vs 第 {t+1} 轮  top-10% 活跃集合 Jaccard = {jac:.3f}")

# 稀疏度拐点：达到 90% / 95% / 99% 覆盖所需 token 比例
print("\n【达到给定覆盖率所需的 token 比例】（决定可用的稀疏度）")
print(f"{'轮次':<6}{'90% 覆盖':>12}{'95% 覆盖':>12}{'99% 覆盖':>12}")
for t in range(1, N_ITER):
    c = np.array(curve_data[t])
    def need(pct):
        idx = np.argmax(c >= pct / 100.0)
        return (idx + 1)
    print(f"{t+1:<6}{need(90):>11}%{need(95):>11}%{need(99):>11}%")

outj = dict(n_scenes=len(scenes), n_tokens=ntok, n_iter=N_ITER, per_iter=report, curves=curve_data)
with open("p0_token_sparsity.json", "w") as f:
    json.dump(outj, f, indent=2)
print("\n结果已存: p0_token_sparsity.json")
