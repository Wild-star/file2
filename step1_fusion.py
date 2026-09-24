#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 融合 POC：给 WAFT-Stereo 注入可训练的【窄带代价体锚】

Step -1 的诊断结论：
  WAFT 编码器(DAv2)的特征不是匹配代价（GT 对齐后 argmax 仅 46-55% 落在正确位置），
  架构里没有任何显式匹配模块 → 迭代 state-dependent → 不可剪枝。
  要加锚，必须【训练】出一个匹配信号。

本实现（最小改动、零初始化、不动已训练权重）：
  ① MatchingBranch : 独立的小卷积网(共享权重)，从【原始图像】提取匹配特征  → 可训练
  ② 窄带代价体     : 在 disp ± R 内做相关，得到 (2R+1) 通道代价
  ③ AnchorProj     : 零初始化 1×1 卷积，把代价聚合成锚特征，【加到】delta_proj 输出上
     → 初始化时锚贡献恒为 0，模型行为 == 原版

训练目标：GT 视差监督（L1 on valid），多轮输出加权求和
"""
import os, sys, math, time, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from bridgedepth.config import get_cfg
from algorithms.waft import WAFT
from model.utils import Padder, disp_warp


# ============================ ①② 匹配分支 + 窄带代价 ============================
class MatchingBranch(nn.Module):
    """从原始图像提取『匹配友好』的特征（独立于 DAv2，可训练）"""
    def __init__(self, ch=32, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 5, stride=2, padding=2), nn.InstanceNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.InstanceNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.InstanceNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch),
        )   # 下采样 8×

    def forward(self, x):
        return self.net(x)


class AnchorHead(nn.Module):
    """把 (2R+1) 通道窄带代价聚合成锚特征，零初始化输出"""
    def __init__(self, n_cost, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(n_cost, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)   # ★ 零初始化

    def forward(self, cost):
        return self.conv(cost)


def narrow_band_cost(m1, m2, disp_lr, R):
    """在 disp_lr ± R 内做相关，返回 (B, 2R+1, h, w)"""
    sims = []
    for o in range(-R, R + 1):
        w = disp_warp(m2, disp_lr + o, padding_mode="zeros")
        sims.append((F.normalize(m1, dim=1) * F.normalize(w, dim=1)).sum(1, keepdim=True))
    return torch.cat(sims, dim=1)


# ============================ 带锚的 WAFT ============================
class WAFTAnchor(WAFT):
    def __init__(self, cfg, R=4, mch=32):
        super().__init__(cfg)
        self.R = R
        self.mb = MatchingBranch(ch=mch, out_ch=mch)
        self.anchor_head = AnchorHead(2 * R + 1, self.hidden_dim)
        self.use_anchor = True

    def forward(self, sample, disp_init=None):
        output = {}
        image1 = self.normalize_image(sample["img1"])
        image2 = self.normalize_image(sample["img2"])
        padder = Padder(image1.shape, factor=self.factor)
        image1p = padder.pad(image1); image2p = padder.pad(image2)

        fmap1, fmap2, net = self.encoder(torch.stack([image1p, image2p], dim=1))

        # 匹配分支：原始图像(未归一化前的 0-255 → 这里用归一化后的即可) → 1/8 分辨率
        if self.use_anchor:
            m1 = self.mb(image1p); m2 = self.mb(image2p)
            # 对齐 fmap 分辨率(1/2)：disp_lr 在 fmap 分辨率，代价也在该分辨率算
            # 为省显存，在 1/8 分辨率算代价，再上采样到 fmap 分辨率
            m_h, m_w = m1.shape[-2:]

        idx_bins_2x = torch.linspace(0, self.max_disp / 2, self.n_bins, device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)
        idx_bins_1x = torch.linspace(0, self.max_disp / 1, self.n_bins, device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)

        prop_hidden = self.prop_proj(torch.cat([fmap1, fmap2], dim=1))
        prop_hidden = self.prop_decoder(prop_hidden)
        prob_mask = .25 * self.prop_mask_head(prop_hidden)
        prob_bins = self.prop_bins_head(prop_hidden)
        prob_up = self.convex_upsample(prob_bins, prob_mask)
        output["init"] = padder.unpad(prob_up)
        pb = F.softmax(prob_bins, dim=1)
        disp = torch.sum(pb * idx_bins_2x, dim=1, keepdim=True)

        delta_disp_preds, delta_info_preds = [], []
        for itr in range(self.iters):
            disp = disp.detach()

            anchor = 0
            if self.use_anchor:
                # disp(fmap 分辨率, 1/2) → 1/8 分辨率（除以 8 的比例关系：fmap 是 1/2，m 是 1/8 → ×1/4）
                disp_m = F.interpolate(disp, size=(m_h, m_w), mode="bilinear", align_corners=True) * 0.25
                cost = narrow_band_cost(m1, m2, disp_m, self.R)             # (B,2R+1,mh,mw)
                anchor = self.anchor_head(cost)
                anchor = F.interpolate(anchor, size=fmap1.shape[-2:], mode="bilinear", align_corners=True)

            warped = disp_warp(fmap2, disp, padding_mode="zeros")
            net = self.delta_proj(torch.cat([fmap1, warped, net, disp], dim=1))
            if self.use_anchor:
                net = net + anchor                                          # ★ 零初始化 → 初始等价原版
            net = self.delta_decoder(net)
            info = self.delta_dist_head(net)
            delta_disp = self.delta_disp_head(net)
            mask = .25 * self.delta_mask_head(net)
            disp = disp + delta_disp
            disp_up = self.convex_upsample(disp * 2, mask)
            info_up = self.convex_upsample(info, mask)
            delta_disp_preds.append(disp_up)
            delta_info_preds.append(info_up)

        output["delta_disp_preds"] = [padder.unpad(p) for p in delta_disp_preds]
        output["delta_info_preds"] = [padder.unpad(p) for p in delta_info_preds]
        output["disp_pred"] = output["delta_disp_preds"][-1].squeeze(1) if self.iters > 0 else \
            torch.sum(F.softmax(output["init"], dim=1) * idx_bins_1x, dim=1)
        return output


def load_pretrained(model, ckpt="ckpts/SynLarge/DAv2S-4.pth"):
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    w = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    w = {k.replace("module.", ""): v for k, v in w.items()}
    miss = model.load_state_dict(w, strict=False)
    return len(w) - len(miss.unexpected_keys), len(miss.missing_keys)
