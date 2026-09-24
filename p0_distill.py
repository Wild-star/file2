#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0② 迭代蒸馏 POC —— 把 3 轮迭代蒸馏进 2 轮（PIP 思路迁移到 WAFT-Stereo）

设计：
  教师 = 3 轮（冻结）    学生 = 2 轮（从教师权重初始化）
  对齐策略（3 轮 → 2 轮，非整除，用"末步对齐"）：
     学生的第 1 轮  ↔  教师的第 1 轮
     学生的第 2 轮  ↔  教师的第 3 轮（即最终输出）  ← 让学生第2步吸收教师的第2+3步
  损失：
     L = ||d1_s - d1_t||²  +  λ · ||d2_s - d3_t||²
  只训练 delta 相关模块（对齐 PIP 的"只 finetune RNN 模块"）
  编码器冻结（本来就是 LoRA + 冻结主干）

数据：ETH3D 27 场景 → 训练 13 / 留出 14
输出：p0_distill_student.pth
"""
import os, sys, json, math, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from bridgedepth.config import get_cfg
from algorithms.waft import WAFT

CFG_T = "configs/SynLarge/DAv2S-4.yaml"
CFG_S = "configs/eval/eth3d_S_iters2.yaml"       # 2 轮配置
CKPT = "ckpts/SynLarge/DAv2S-4.pth"
DATA = "datasets/ETH3D/two_view_training"
STEPS = int(os.environ.get("STEPS", "1200"))
BATCH = int(os.environ.get("BATCH", "1"))
CROP = (256, 384)
LR = 1e-5
LAMBDA = 4.0                # 末步对齐权重
SEED = 0
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

dev = "cuda"


def build(cfgfile):
    cfg = get_cfg(); cfg.merge_from_file(cfgfile)
    m = WAFT(cfg).to(dev)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    w = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    w = {k.replace("module.", ""): v for k, v in w.items()}
    m.load_state_dict(w, strict=False)
    return m, cfg


teacher, cfg_t = build(CFG_T)
student, cfg_s = build(CFG_S)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# 只训练 delta 模块 + 相关投影/头（对齐 PIP：只 finetune recurrent 模块）
TRAIN_KEYS = ("delta_decoder", "delta_proj", "delta_disp_head", "delta_dist_head", "delta_mask_head")
for n, p in student.named_parameters():
    p.requires_grad = any(k in n for k in TRAIN_KEYS)
n_tr = sum(p.numel() for p in student.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in student.parameters())
print(f"[学生] 可训练 {n_tr/1e6:.3f} M / 总 {n_all/1e6:.3f} M")
print(f"[教师] {len(cfg_t.WAFT.ITERATIVE_MODULE.TASK)} 轮 | [学生] {len(cfg_s.WAFT.ITERATIVE_MODULE.TASK)} 轮")

opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)

# ---------------- 数据 ----------------
scenes = sorted(x for x in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, x)))
random.Random(SEED).shuffle(scenes)
n_tr_sc = int(len(scenes) * 0.5)
train_scenes, holdout_scenes = scenes[:n_tr_sc], scenes[n_tr_sc:]
print(f"[数据] 训练场景 {len(train_scenes)} | 留出场景 {len(holdout_scenes)}")
json.dump(dict(train=train_scenes, holdout=holdout_scenes), open("p0_split.json", "w"), indent=2)


class SceneSet:
    def __init__(self, names): self.names = names
    def sample(self):
        sc = random.choice(self.names)
        d = os.path.join(DATA, sc)
        a = np.asarray(Image.open(os.path.join(d, "im0.png")).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(os.path.join(d, "im1.png")).convert("RGB"), dtype=np.float32)
        t0 = torch.from_numpy(a).permute(2, 0, 1)
        t1 = torch.from_numpy(b).permute(2, 0, 1)
        _, H, W = t0.shape
        ch, cw = CROP
        if H < ch or W < cw:      # 太小的图整张用
            ch, cw = (H // 16) * 16, (W // 16) * 16
        y = random.randint(0, max(0, H - ch)); x = random.randint(0, max(0, W - cw))
        return t0[:, y:y+ch, x:x+cw][None].to(dev), t1[:, y:y+ch, x:x+cw][None].to(dev)


trainset = SceneSet(train_scenes)

# ---------------- 训练 ----------------
# 需要拿到"每轮 disp"：forward 已返回 delta_disp_preds（list）
print(f"\n[训练] {STEPS} 步, batch={BATCH}, crop={CROP}, lr={LR}, λ={LAMBDA}\n")
t0 = time.time()
hist = []
for step in range(1, STEPS + 1):
    x1 = []; x2 = []
    for _ in range(BATCH):
        a, b = trainset.sample(); x1.append(a[0]); x2.append(b[0])
    sample = {"img1": torch.stack(x1), "img2": torch.stack(x2)}

    with torch.no_grad():
        out_t = teacher(sample)
        dt = out_t["delta_disp_preds"]                 # [d1, d2, d3]
    out_s = student(sample)
    ds = out_s["delta_disp_preds"]                     # [d1, d2]

    # 学生的第1轮 ↔ 教师第1轮；学生第2轮 ↔ 教师末轮
    l1 = F.mse_loss(ds[0], dt[0])
    l2 = F.mse_loss(ds[-1], dt[-1])
    loss = l1 + LAMBDA * l2

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    opt.step()

    if step % 100 == 0 or step == 1:
        el = time.time() - t0
        print(f"  step {step:>5}/{STEPS}  loss={loss.item():.6f}  L1={l1.item():.6f}  L2={l2.item():.6f}  {el/step*1000:.0f} ms/step", flush=True)
        hist.append(dict(step=step, loss=loss.item(), l1=l1.item(), l2=l2.item()))

json.dump(hist, open("p0_distill_history.json", "w"), indent=2)
torch.save({"model": {k: v.cpu() for k, v in student.state_dict().items()}},
           "p0_distill_student.pth")
print(f"\n[完成] 用时 {(time.time()-t0)/60:.1f} 分钟 → p0_distill_student.pth")
