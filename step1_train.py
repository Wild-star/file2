#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 训练：只训练【新增的锚模块】（MatchingBranch + AnchorHead）
              基础模型完全冻结不动 → 干净的单变量消融

数据：WMGStereo indoor（595 对，含 GT 视差 + 有效掩码）
目标：GT 视差监督（L1 on valid），多轮输出加权
产物：step1_anchor.pth
"""
import os, sys, glob, math, time, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from bridgedepth.config import get_cfg
from step1_fusion import WAFTAnchor, load_pretrained

CFG = "configs/SynLarge/DAv2S-4.yaml"
CKPT = "ckpts/SynLarge/DAv2S-4.pth"
DATA = "datasets/WMGStereo"
OUT = os.environ.get("OUT", "step1_anchor.pth")
STEPS = int(os.environ.get("STEPS", "1500"))
BATCH = int(os.environ.get("BATCH", "2"))
CROP = (320, 448)
LR = float(os.environ.get("LR", "3e-4"))
R = int(os.environ.get("ANC_R", "4"))
SEED = 0
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
dev = "cuda"


def build_pairs(root):
    """WMGStereo 命名约定：
        左: Image/camera_0/Image_<stem>_0.png
        右: Image/camera_1/Image_<stem>_1.png      ← 注意后缀是 _1
        视差: disparity/camera_0/disparity_<stem>_0.npy
        掩码: disparity_masks/camera_0/valid_<stem>_0.png
    """
    pairs = []
    for seed in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        fr = os.path.join(root, seed, "frames")
        for df in sorted(glob.glob(os.path.join(fr, "disparity/camera_0/*.npy"))):
            stem = os.path.basename(df)[len("disparity_"):-len(".npy")]
            if stem.endswith("_0"):
                stem = stem[:-2]
            l = os.path.join(fr, f"Image/camera_0/Image_{stem}_0.png")
            r = os.path.join(fr, f"Image/camera_1/Image_{stem}_1.png")
            m = os.path.join(fr, f"disparity_masks/camera_0/valid_{stem}_0.png")
            if os.path.exists(l) and os.path.exists(r) and os.path.exists(m):
                pairs.append((l, r, df, m))
    return pairs


pairs = build_pairs(DATA)
print(f"[数据] {len(pairs)} 对立体像对")

cfg = get_cfg(); cfg.merge_from_file(CFG)
model = WAFTAnchor(cfg, R=R).to(dev).train()
n, miss = load_pretrained(model, CKPT)
print(f"[模型] 载入 {n} 枚预训练权重")

# ★ 冻结全部，只放新增模块
MODE = os.environ.get("TRAIN_MODE", "anchor")
for p in model.parameters():
    p.requires_grad = False
for mod in (model.mb, model.anchor_head):
    for p in mod.parameters():
        p.requires_grad = True
if MODE == "joint":
    JOINT_KEYS = ("delta_decoder", "delta_proj", "delta_disp_head", "delta_dist_head", "delta_mask_head")
    for n_, p in model.named_parameters():
        if any(k in n_ for k in JOINT_KEYS):
            p.requires_grad = True
print(f"[模式] TRAIN_MODE={MODE}")
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[训练] 可训练参数 {n_tr/1e3:.1f} K （仅 MatchingBranch + AnchorHead）")

print("[验证] 零初始化是否等价原版...")
with torch.no_grad():
    l, r, df, mf = pairs[0]
    a = torch.from_numpy(np.asarray(Image.open(l).convert("RGB"))).permute(2, 0, 1).float()[None].cuda()
    b = torch.from_numpy(np.asarray(Image.open(r).convert("RGB"))).permute(2, 0, 1).float()[None].cuda()
    o1 = model({"img1": a, "img2": b})["disp_pred"].clone()
    model.use_anchor = False
    o0 = model({"img1": a, "img2": b})["disp_pred"].clone()
    model.use_anchor = True
    d = (o1 - o0).abs().max().item()
    print(f"      有锚 vs 无锚 最大差 = {d:.3e}  {'✅ 等价（零初始化生效）' if d < 1e-4 else '❌ 不等价'}")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)


def sample_batch():
    xs1, xs2, gts, vals = [], [], [], []
    for _ in range(BATCH):
        l, r, df, mf = random.choice(pairs)
        a = np.asarray(Image.open(l).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(r).convert("RGB"), dtype=np.float32)
        g = np.load(df).astype(np.float32)
        v = (np.asarray(Image.open(mf)) > 0) & (g > 0) & (g < 1e3)
        H, W = a.shape[:2]
        ch, cw = CROP
        if H < ch or W < cw: ch, cw = (H // 8) * 8, (W // 8) * 8
        y = random.randint(0, max(0, H - ch)); x = random.randint(0, max(0, W - cw))
        sl = (slice(y, y + ch), slice(x, x + cw))
        xs1.append(torch.from_numpy(a[sl]).permute(2, 0, 1))
        xs2.append(torch.from_numpy(b[sl]).permute(2, 0, 1))
        gts.append(torch.from_numpy(g[sl])); vals.append(torch.from_numpy(v[sl]))
    return (torch.stack(xs1).to(dev), torch.stack(xs2).to(dev),
            torch.stack(gts).to(dev), torch.stack(vals).to(dev))


print(f"\n[训练] {STEPS} 步 | batch {BATCH} | crop {CROP} | lr {LR} | R={R}\n")
t0 = time.time(); hist = []
for step in range(1, STEPS + 1):
    x1, x2, gt, val = sample_batch()
    out = model({"img1": x1, "img2": x2})
    preds = out["delta_disp_preds"]
    loss = 0
    for i, p in enumerate(preds):
        w = 0.5 ** (len(preds) - 1 - i)
        p = p.squeeze(1)
        e = (p - gt).abs()
        loss = loss + w * (e[val].mean() if val.any() else e.mean())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step(); sched.step()
    if step % 100 == 0 or step == 1:
        el = time.time() - t0
        print(f"  step {step:>5}/{STEPS}  loss={loss.item():.4f}  {el/step*1000:.0f} ms/step", flush=True)
        hist.append(dict(step=step, loss=loss.item()))

torch.save({"anchor": {k: v.cpu() for k, v in model.state_dict().items()
                       if k.startswith("mb.") or k.startswith("anchor_head.")},
            "joint": {k: v.cpu() for k, v in model.state_dict().items()
                      if not k.startswith("mb.") and not k.startswith("anchor_head.")},
            "R": R, "hist": hist, "mode": MODE}, OUT)
json.dump(hist, open("step1_hist.json", "w"), indent=2)
print(f"\n[完成] {(time.time()-t0)/60:.1f} 分钟 → {OUT}")
