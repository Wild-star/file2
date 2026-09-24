#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 — FusionWarp-Stereo 复合脚本（paper design 的可运行 POC）

把调研到的近 5 年「立体匹配 / 光流」模块融合进 WAFT 式『warp 迭代 + 场变换』框架：

  ① GlobalMatcher      ← GMFlow / STTR / FlowFormer 的 transformer 全局匹配
                         （全视差范围、1/8 粗粒度，直接回归初始视差 → 同 SEA-RAFT 的 direct initial flow）
  ② SparseCorrAnchor   ← RAFT-Stereo / GwcNet / CREStereo 的 group-wise 窄带相关锚
                         （在 disp±R 内做分组相关，注入迭代更新，零初始化 → 手术安全）
  ②b GEVCostAnchor     ← PSMNet / GwcNet / IGEV 的「传统代价体」轻量引用（ANCHOR_KIND=gev）
                         （1/4 尺度窄带组合几何编码体 + 轻量 3D 正则化聚合）
  ③ GatedFusion        ← ACVNet patch-attention / CREStereo 注意力的可学习门控融合
  ④ TokenSparsity      ← 本仓库 P0 诊断 + Selective-Stereo 多频选择思想的 token 级稀疏更新
  ⑤ MixtureLaplaceLoss ← SEA-RAFT 的混合拉普拉斯鲁棒损失

本脚本为【自包含】实现（仅 torch / numpy），镜像 WAFT 的核心算子（disp_warp、
convex_upsample、逐轮迭代 + 场变换），在 CPU 上跑通：合成立体对 → 前向 →
反向 → N 步训练 → 打印 EPE / bad-1px / token 稀疏度，并自检收敛。

用法：
    python step2_fusion_composite.py                 # 默认 120 步训练（corr 锚）
    ANCHOR_KIND=gev python step2_fusion_composite.py # 换用窄带代价体 + 3D 聚合锚
    STEPS=300 BATCH=6 python step2_fusion_composite.py
    ABLATE=1 python step2_fusion_composite.py        # 额外打印逐模块消融（输出是否变化）
"""
import os, sys, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 0. 基础算子（镜像 WAFT 的 model/utils.py）
# --------------------------------------------------------------------------- #
def meshgrid(img):
    b, _, h, w = img.size()
    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)
    grid = torch.cat((x_range, y_range), dim=0)
    return grid.unsqueeze(0).expand(b, 2, h, w)


def normalize_coords(grid):
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :] / (w - 1)) - 1
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :] / (h - 1)) - 1
    return grid.permute((0, 2, 3, 1))


def disp_warp(feature, disp, padding_mode='border'):
    """按 -disp 沿水平方向 warp（右图特征对齐到左图坐标）。"""
    grid = meshgrid(feature)
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)
    return F.grid_sample(feature, sample_grid, mode='bilinear',
                         padding_mode=padding_mode, align_corners=True)


def convex_upsample(info, mask):
    """RAFT/WAFT 的 convex upsampling（2x，9 邻域权重）。"""
    N, C, H, W = info.shape
    mask = mask.view(N, 1, 9, 2, 2, H, W)
    mask = torch.softmax(mask, dim=2)
    up = F.unfold(info, [3, 3], padding=1)
    up = up.view(N, C, 9, 1, 1, H, W)
    up = torch.sum(mask * up, dim=2)
    up = up.permute(0, 1, 4, 2, 5, 3)
    return up.reshape(N, C, 2 * H, 2 * W)


def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)


def res_block(c, stride=1):
    return nn.Sequential(
        conv3x3(c, c, stride), nn.GroupNorm(8, c), nn.SiLU(inplace=True),
        conv3x3(c, c, 1), nn.GroupNorm(8, c))


# --------------------------------------------------------------------------- #
# 1. 特征编码器（context 特征）+ 匹配分支（matching 友好特征）
# --------------------------------------------------------------------------- #
class FeatureEncoder(nn.Module):
    """极小 CNN，stride=2 → 1/2 分辨率，输出 C 通道 context 特征。"""
    def __init__(self, C=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, C, 3, 2, 1, bias=False), nn.GroupNorm(8, C), nn.SiLU(inplace=True),
            res_block(C),
            nn.Conv2d(C, C, 3, 1, 1, bias=False), nn.GroupNorm(8, C), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class MatchFeat(nn.Module):
    """独立可训练匹配分支（Step-1 结论：context 特征非匹配代价，需单独分支）。"""
    def __init__(self, C=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, C, 5, 2, 2, bias=False), nn.InstanceNorm2d(C), nn.SiLU(inplace=True),
            nn.Conv2d(C, C, 3, 1, 1, bias=False), nn.InstanceNorm2d(C), nn.SiLU(inplace=True),
            nn.Conv2d(C, C, 3, 1, 1))

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# 2. GlobalMatcher（GMFlow / STTR / FlowFormer 的全局匹配）
# --------------------------------------------------------------------------- #
def mha(q, k, v, heads):
    """多头注意力，q/k/v: (B, N, C)。"""
    B, N, C = q.shape
    d = C // heads
    q = q.view(B, N, heads, d).transpose(1, 2)
    k = k.view(B, N, heads, d).transpose(1, 2)
    v = v.view(B, N, heads, d).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(B, N, C)


class CrossAttentionLayer(nn.Module):
    """左↔右 token 交叉注意力（全局上下文增强，处理低纹理/大视差）。"""
    def __init__(self, C, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.heads = heads
        self.ff = nn.Sequential(nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, left, right):
        l2 = mha(self.norm1(left), self.norm1(right), self.norm1(right), self.heads) + left
        l2 = self.ff(self.norm2(l2)) + l2
        r2 = mha(self.norm1(right), self.norm1(left), self.norm1(left), self.heads) + right
        r2 = self.ff(self.norm2(r2)) + r2
        return l2, r2


class GlobalMatcher(nn.Module):
    """
    在 1/8 粗粒度做全视差范围相关 + soft-argmax，直接回归初始视差 d_gm；
    同时输出全局上下文特征 g_feat 作为迭代隐状态的种子。
    """
    def __init__(self, C=32, D_bins=9, heads=4, img_h=96, img_w=128):
        super().__init__()
        self.C = C
        self.D_bins = D_bins
        self.h, self.w = img_h // 8, img_w // 8          # 1/8 token 网格
        N = self.h * self.w
        self.pos = nn.Parameter(torch.randn(1, N, C) * 0.02)
        self.cross = CrossAttentionLayer(C, heads)
        self.proj = nn.Conv2d(2 * C, C, 1)
        self.disp_idx = torch.arange(D_bins).view(1, D_bins, 1, 1).float()

    def forward(self, f1, f2):
        B = f1.shape[0]
        # 1/2 → 1/8
        f1c = F.avg_pool2d(f1, 4, 4)
        f2c = F.avg_pool2d(f2, 4, 4)
        # → tokens，加位置编码，交叉注意力
        l = f1c.flatten(2).transpose(1, 2) + self.pos
        r = f2c.flatten(2).transpose(1, 2) + self.pos
        l, r = self.cross(l, r)
        l = l.transpose(1, 2).reshape(B, self.C, self.h, self.w)
        r = r.transpose(1, 2).reshape(B, self.C, self.h, self.w)
        # 全视差范围相关 + soft-argmax（1/8 尺度）
        l = F.normalize(l, dim=1)
        cost = []
        disp_idx = self.disp_idx.to(f1.device)
        for d in range(self.D_bins):
            w = disp_warp(r, torch.full((B, 1, self.h, self.w), float(d), device=f1.device),
                          padding_mode='zeros')
            cost.append((l * F.normalize(w, dim=1)).sum(1, keepdim=True))
        cost = torch.cat(cost, dim=1)                       # (B, D_bins, h, w)
        prob = F.softmax(cost, dim=1)
        d_gm = torch.sum(prob * disp_idx, dim=1, keepdim=True)   # 1/8 尺度视差
        # 上采样到 1/2 尺度：d_gm ×4，g_feat ×4
        d_gm = F.interpolate(d_gm, scale_factor=4, mode='bilinear', align_corners=True) * 4.0
        g_feat = F.interpolate(torch.cat([l, r], dim=1), scale_factor=4,
                               mode='bilinear', align_corners=True)  # (B, 2C, H/2, W/2)
        g_feat = self.proj(g_feat)                                   # (B, C, H/2, W/2)
        return d_gm, g_feat


# --------------------------------------------------------------------------- #
# 3. SparseCorrAnchor（group-wise 窄带相关锚）
# --------------------------------------------------------------------------- #
def group_corr(m1, m2, G):
    B, C, h, w = m1.shape
    Cg = C // G
    a = F.normalize(m1.view(B, G, Cg, h, w), dim=2)
    b = F.normalize(m2.view(B, G, Cg, h, w), dim=2)
    return (a * b).sum(dim=2)                                # (B, G, h, w)


class SparseCorrAnchor(nn.Module):
    """在 disp±R 内做 group-wise 相关，聚合成锚特征；零初始化 → 手术安全。"""
    def __init__(self, C=32, G=8, R=4, out_ch=32):
        super().__init__()
        self.G, self.R, self.out_ch = G, R, out_ch
        self.head = nn.Conv2d((2 * R + 1) * G, out_ch, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, m1, m2, disp):
        sims = []
        for o in range(-self.R, self.R + 1):
            w = disp_warp(m2, disp + o, padding_mode='zeros')
            sims.append(group_corr(m1, w, self.G))
        cost = torch.cat(sims, dim=1)                        # (B, (2R+1)*G, h, w)
        return self.head(cost)


# --------------------------------------------------------------------------- #
# 3b. GEVCostAnchor（对「传统代价体」的轻量引用）
# --------------------------------------------------------------------------- #
class GEVCostAnchor(nn.Module):
    """
    传统代价体（PSMNet/GwcNet/IGEV）的「轻量引用」：在 1/4 粗尺度构造
    『组合几何编码体』= group-wise 相关（匹配证据）+ 当前视差（几何信息），
    经 2 层轻量 3D 卷积在「视差维 × 空间维」聚合后：
      - prob = softmax_k(cost) → d_cv = Σ prob·d_k     （代价体先验视差，可监督）
      - feat = Σ prob·agg                            （概率加权的代价体上下文特征，作锚）

    与 SparseCorrAnchor 的区别：前者是【无聚合的逐点相关】；这里是【3D 卷积正则化聚合】，
    正是传统代价体的核心价值（邻域 + 跨视差平滑、可学习代价滤波）。
    只在 1/4 尺度 + 窄带 + 组相关 → 内存/计算可控，不违背 WAFT 高效初衷。
    """
    def __init__(self, C=32, G=4, K=9, R=8, Cv=8, out_ch=32, agg_kind='full3d', downsample=2):
        super().__init__()
        self.G, self.K, self.R, self.downsample = G, K, R, downsample
        # 非均匀采样偏移：中心密、边缘疏（Selective-Stereo 稀疏采样思想）
        t = torch.linspace(-1, 1, K)
        offs = (R * torch.sign(t) * t.abs() ** 2).tolist()
        offs[0], offs[-1] = -float(R), float(R)
        self.offsets = offs
        self.agg_kind = agg_kind
        self.agg = self._build_agg(G + 1, Cv)
        self.cost_head = nn.Conv3d(Cv, 1, (1, 1, 1))
        self.feat_proj = nn.Conv2d(Cv, out_ch, 3, padding=1)
        # 零初始化输出 → 手术安全：feat=0（锚等价原版），d_cv=disp_q（中性先验，不污染初始视差）
        for m in (self.cost_head, self.feat_proj):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _sep_block(self, cin, cout):
        # 可分离 3D：空间 2D 聚合 (1,3,3) + 视差维 1D 聚合 (3,1,1) —— PSMNet/GA-Net 轻量化思想
        return nn.Sequential(
            nn.Conv3d(cin, cout, (1, 3, 3), padding=(0, 1, 1)), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0)), nn.ReLU(inplace=True))

    def _build_agg(self, cin, Cv):
        if self.agg_kind == 'sep3d':
            return nn.Sequential(self._sep_block(cin, Cv), self._sep_block(Cv, Cv))
        return nn.Sequential(  # full3d：完整 (3,3,3) 卷积
            nn.Conv3d(cin, Cv, (3, 3, 3), padding=(1, 1, 1)), nn.ReLU(inplace=True),
            nn.Conv3d(Cv, Cv, (3, 3, 3), padding=(1, 1, 1)), nn.ReLU(inplace=True))

    def forward(self, m1, m2, disp):
        # 可选下采样（默认 ×2 → 1/4）；视差值也换算到对应尺度单位（×1/downsample）
        ds = self.downsample
        if ds > 1:
            m1 = F.avg_pool2d(m1, ds, ds)
            m2 = F.avg_pool2d(m2, ds, ds)
            disp_q = F.avg_pool2d(disp, ds, ds) * (1.0 / ds)
        else:
            disp_q = disp
        B, C, h, w = m1.shape
        Cg = C // self.G
        corrs, geos = [], []
        for o in self.offsets:
            d = disp_q + o
            w2 = disp_warp(m2, d, padding_mode='zeros')
            g = (F.normalize(m1.view(B, self.G, Cg, h, w), dim=2) *
                 F.normalize(w2.view(B, self.G, Cg, h, w), dim=2)).sum(dim=2)   # (B,G,h,w)
            corrs.append(g)
            geos.append(d)
        vol = torch.cat([torch.stack(corrs, dim=2), torch.stack(geos, dim=2)], dim=1)  # (B,G+1,K,h,w)
        agg = self.agg(vol)                                     # (B,Cv,K,h,w)
        cost = self.cost_head(agg).squeeze(1)                   # (B,K,h,w)
        prob = F.softmax(cost, dim=1)
        dk = torch.stack([disp_q + o for o in self.offsets], dim=1).squeeze(2)  # (B,K,h,w)
        d_cv = (prob * dk).sum(dim=1, keepdim=True)             # (B,1,h,w) @1/4
        feat = (prob.unsqueeze(1) * agg).sum(dim=2)             # (B,Cv,h,w)
        feat = self.feat_proj(feat)                             # (B,out_ch,h,w)
        d_cv = F.interpolate(d_cv, scale_factor=ds, mode='bilinear', align_corners=True) * float(ds)
        feat = F.interpolate(feat, scale_factor=ds, mode='bilinear', align_corners=True)
        return d_cv, feat                                       # 均回到 1/2 尺度


# --------------------------------------------------------------------------- #
# 4. GatedFusion（可学习门控融合）
# --------------------------------------------------------------------------- #
class GatedFusion(nn.Module):
    """空间门控：gate=σ(conv([anchor, g_feat]))，融合锚与全局上下文。"""
    def __init__(self, C=32):
        super().__init__()
        self.gate = nn.Conv2d(2 * C, 1, 3, padding=1)

    def forward(self, anchor, g_feat):
        g = torch.sigmoid(self.gate(torch.cat([anchor, g_feat], dim=1)))
        return g * anchor + (1 - g) * g_feat


# --------------------------------------------------------------------------- #
# 5. TokenSparseViT（带 token 稀疏更新的迭代解码器）
# --------------------------------------------------------------------------- #
class TokenSparseViT(nn.Module):
    """
    token 级稀疏更新：router 预测每个 token 的 saliency，gate 决定「更新 vs 保留」。
    硬 top-k 版本在完整训练中省算力，此处用可微软门控跑通并统计稀疏度。
    """
    def __init__(self, dim=32, depth=2, heads=4, patch=8, out_ch=32, sparse=True):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.sparse = sparse
        self.patch_embed = nn.Conv2d(dim, dim, patch, stride=patch)
        self.sal_conv = nn.Conv2d(dim, 1, patch, stride=patch)          # 与 token 网格对齐
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(dim),
                attn_qkv=nn.Linear(dim, 3 * dim),
                n2=nn.LayerNorm(dim),
                mlp=nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)))))
        self.heads = heads
        self.res = nn.Sequential(res_block(dim), res_block(dim))
        self.final = nn.Conv2d(2 * dim, out_ch, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        tok = self.patch_embed(x)                              # (B, C, Th, Tw)
        Th, Tw = tok.shape[-2:]
        tok = tok.flatten(2).transpose(1, 2)                   # (B, N, C)
        sal = self.sal_conv(x).flatten(2).transpose(1, 2)      # (B, N, 1)
        gate = torch.sigmoid(sal)                              # 每 token 更新幅度
        if not self.sparse:
            gate = torch.ones_like(gate)

        h = tok
        for blk in self.blocks:
            n = blk['n1'](h)
            qkv = blk['attn_qkv'](n)
            q, k, v = qkv.chunk(3, dim=-1)
            Bn, Nn, Cn = q.shape
            d = Cn // self.heads
            q = q.view(Bn, Nn, self.heads, d).transpose(1, 2)
            k = k.view(Bn, Nn, self.heads, d).transpose(1, 2)
            v = v.view(Bn, Nn, self.heads, d).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(Bn, Nn, Cn)
            h = h + a
            h = h + blk['mlp'](blk['n2'](h))

        # 稀疏更新：gate→0 保留原 token，gate→1 完整更新
        updated = tok + gate * (h - tok)
        updated = updated.transpose(1, 2).reshape(B, C, Th, Tw)
        updated = F.interpolate(updated, size=(H, W), mode='bilinear', align_corners=True)
        res = self.res(x)
        out = self.final(torch.cat([updated, res], dim=1))
        return out, gate.mean()


# --------------------------------------------------------------------------- #
# 6. 损失（SEA-RAFT 混合拉普拉斯）
# --------------------------------------------------------------------------- #
def mixture_laplace_nll(err, pi=0.5, b0=0.05, b1=0.5):
    log_p0 = -math.log(2 * b0) - err / b0
    log_p1 = -math.log(2 * b1) - err / b1
    lp0 = math.log(pi) + log_p0
    lp1 = math.log(1.0 - pi) + log_p1
    return -torch.logsumexp(torch.stack([lp0, lp1], dim=0), dim=0).mean()


# --------------------------------------------------------------------------- #
# 7. 顶层模型：FusionWarp-Stereo
# --------------------------------------------------------------------------- #
class FusionWarpStereo(nn.Module):
    def __init__(self, C=32, hidden=32, G=8, R=4, D_bins=9, iters=2,
                 img_h=96, img_w=128, use_anchor=True, use_sparse=True,
                 anchor_kind='corr', agg_kind='full3d', gev_downsample=2, gev_R=8):
        super().__init__()
        self.encoder = FeatureEncoder(C)
        self.match = MatchFeat(C)
        self.global_matcher = GlobalMatcher(C, D_bins, img_h=img_h, img_w=img_w)
        self.fusion = GatedFusion(C)
        self.delta_proj = nn.Conv2d(4 * C + 1, hidden, 3, padding=1)
        self.delta_decoder = TokenSparseViT(hidden, depth=2, heads=4, patch=8, out_ch=hidden,
                                            sparse=use_sparse)
        self.disp_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.mask_head = nn.Conv2d(hidden, 9 * 4, 3, padding=1)
        self.iters = iters
        self.use_anchor = use_anchor
        self.use_sparse = use_sparse
        self.anchor_kind = anchor_kind
        # 锚模块最后构造 → 共享模块（encoder/match/global/fusion/delta/heads）在
        # corr 与 gev 之间得到一致的随机初始化，保证消融公平（只差锚本身）
        if anchor_kind == 'gev':
            self.gev_anchor = GEVCostAnchor(C, G=4, K=9, R=gev_R, Cv=8, out_ch=hidden,
                                            agg_kind=agg_kind, downsample=gev_downsample)
        else:
            self.anchor = SparseCorrAnchor(C, G, R, hidden)

    def forward(self, img1, img2):
        f1, f2 = self.encoder(img1), self.encoder(img2)         # 1/2
        m1, m2 = self.match(img1), self.match(img2)             # 1/2
        d_gm, g_feat = self.global_matcher(f1, f2)              # d_gm(1/2), g_feat(1/2, C)
        disp = d_gm                                             # 全局初始视差
        net = None
        preds, gates = [], []
        for _ in range(self.iters):
            disp = disp.detach()
            d_cv = None
            if self.use_anchor:
                if self.anchor_kind == 'gev':
                    d_cv, anchor = self.gev_anchor(m1, m2, disp)
                else:
                    anchor = self.anchor(m1, m2, disp)          # 零初始化
            else:
                anchor = torch.zeros_like(disp).repeat(1, f1.shape[1], 1, 1)
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
            disp_up = convex_upsample(disp * 2, mask)           # 1/2 → 全分辨率
            preds.append(disp_up)
        return {'preds': preds, 'gates': gates, 'd_gm': d_gm, 'd_cv': d_cv}


# --------------------------------------------------------------------------- #
# 8. 合成立体对
# --------------------------------------------------------------------------- #
def make_synthetic_batch(B, H, W, max_disp=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    lefts, disps = [], []
    for _ in range(B):
        # 平滑多斑点纹理（低纹理 + 一定纹理，考验全局匹配）
        z = torch.randn(1, 3, H // 8, W // 8, generator=g)
        tex = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=True)
        noise = torch.randn(1, 3, H, W, generator=g) * 0.06
        left = torch.sigmoid(tex * 1.5) * 0.7 + 0.3 + noise
        left = left.clamp(0, 1)
        # 平滑视差场：随机平面 + 正弦涟漪
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing='ij')
        d0 = 4 + torch.rand(1, generator=g).item() * max_disp * 0.4
        gx = (torch.rand(1, generator=g).item() - 0.5) * max_disp * 0.6
        gy = (torch.rand(1, generator=g).item() - 0.5) * max_disp * 0.6
        amp = torch.rand(1, generator=g).item() * 6
        fx = 2 + torch.rand(1, generator=g).item() * 4
        disp = d0 + gx * (xx / W) + gy * (yy / H) + amp * torch.sin(fx * 2 * math.pi * xx / W)
        disp = disp.clamp(1, max_disp).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        lefts.append(left)
        disps.append(disp)
    left = torch.cat(lefts, 0)                                     # (B,3,H,W)
    gt = torch.cat(disps, 0)                                       # (B,1,H,W)
    right = disp_warp(left, gt, padding_mode='border')             # 右图 = 左图按 -disp 平移
    valid = torch.ones_like(gt).bool()
    return left, right, gt, valid


# --------------------------------------------------------------------------- #
# 9. 主流程
# --------------------------------------------------------------------------- #
def main():
    STEPS = int(os.environ.get('STEPS', '120'))
    BATCH = int(os.environ.get('BATCH', '4'))
    H, W = 96, 128
    MAX_DISP = 64
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    anchor_kind = os.environ.get('ANCHOR_KIND', 'corr')   # 'corr' | 'gev'
    agg_kind = os.environ.get('AGG_KIND', 'full3d')       # 'full3d' | 'sep3d'（仅 gev）
    print(f"[设备] {dev}  |  STEPS={STEPS}  BATCH={BATCH}  input={H}x{W}  max_disp={MAX_DISP}  "
          f"anchor_kind={anchor_kind}  agg_kind={agg_kind}")

    model = FusionWarpStereo(img_h=H, img_w=W, anchor_kind=anchor_kind,
                             agg_kind=agg_kind).to(dev).train()
    n_params = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 参数量 {n_params/1e3:.1f}K（可训练 {n_tr/1e3:.1f}K）")

    # 自检：corr 锚零初始化 → anchor 输出应为 0（手术安全）；gev 锚带 3D 聚合，非零初始化
    if anchor_kind == 'corr':
        with torch.no_grad():
            x1 = torch.randn(1, 3, H, W)
            z = model.anchor(model.match(x1), model.match(x1), torch.zeros(1, 1, H // 2, W // 2))
            print(f"[自检] 零初始化锚输出 max|anchor| = {z.abs().max().item():.3e}  "
                  f"{'✅ 等价原版（零初始化生效）' if z.abs().max().item() < 1e-5 else '❌'}")
            del z
    else:
        with torch.no_grad():
            x1 = torch.randn(1, 3, H, W)
            mf = model.match(x1)
            d0 = torch.zeros(1, 1, H // 2, W // 2)
            d_cv, feat = model.gev_anchor(mf, mf, d0)
            okz = feat.abs().max().item() < 1e-5 and (d_cv - d0).abs().max().item() < 1e-5
            print(f"[自检] gev 锚零初始化  max|feat|={feat.abs().max().item():.3e}  "
                  f"d_cv 偏离当前视差={(d_cv - d0).abs().max().item():.3e}  "
                  f"{'✅ 手术安全（等价原版）' if okz else '❌'}")
            del d_cv, feat

    left, right, gt, valid = make_synthetic_batch(BATCH, H, W, MAX_DISP, seed=SEED)
    left, right, gt, valid = [t.to(dev) for t in (left, right, gt, valid)]
    print(f"[数据] 合成立体对 {BATCH} 对，GT 视差范围 [{gt.min().item():.1f}, {gt.max().item():.1f}] px")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.1)

    print(f"\n[训练] {STEPS} 步（前向 + 反向，混合拉普拉斯损失）\n")
    hist = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        out = model(left, right)
        preds = out['preds']
        loss = 0.0
        # 全局初始视差的辅助监督（SEA-RAFT direct initial flow 思想；d_gm 在 1/2 尺度）
        gt_half = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5
        loss = loss + 0.5 * mixture_laplace_nll((out['d_gm'] - gt_half).abs())
        # 代价体先验视差 d_cv 的辅助监督（仅 gev 锚）
        if out.get('d_cv') is not None:
            loss = loss + 0.3 * mixture_laplace_nll((out['d_cv'] - gt_half).abs())
        for i, p in enumerate(preds):
            w = 0.5 ** (len(preds) - 1 - i)
            err = (p - gt).abs()
            loss = loss + w * mixture_laplace_nll(err[valid])
        sparsity = torch.stack(out['gates']).mean()
        loss = loss + 0.02 * sparsity                       # 鼓励 token 稀疏
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                epe = (preds[-1] - gt).abs()[valid].mean().item()
                bad1 = ((preds[-1] - gt).abs() > 1.0)[valid].float().mean().item()
            hist.append(dict(step=step, loss=loss.item(), epe=epe, bad1=bad1,
                             sparsity=sparsity.item()))
            print(f"  step {step:>4}/{STEPS}  loss={loss.item():.4f}  EPE={epe:.3f}px  "
                  f"bad1px={bad1*100:.1f}%  gate(稀疏度)={sparsity.item():.3f}", flush=True)

    dt = time.time() - t0
    epe0 = hist[0]['epe']
    epe1 = hist[-1]['epe']
    print(f"\n[结果] 耗时 {dt:.1f}s  |  EPE {epe0:.3f} → {epe1:.3f} px  "
          f"| bad1px {hist[-1]['bad1']*100:.1f}%")

    ok = epe1 < 0.6 * epe0
    print(f"[自检] 收敛判定（末步 EPE < 0.6×首步 EPE）：{'✅ PASS' if ok else '❌ FAIL'}")
    with open('step2_hist.json', 'w') as f:
        json.dump(dict(steps=STEPS, batch=BATCH, epe_first=epe0, epe_last=epe1, hist=hist), f, indent=2)
    print("[输出] 指标已存 step2_hist.json")

    # 训练后逐模块消融（可选）：每个模块都应当改变了最终输出
    if os.environ.get('ABLATE', '0') == '1':
        model.eval()
        with torch.no_grad():
            base = model(left, right)['preds'][-1]
            m_a = FusionWarpStereo(img_h=H, img_w=W, anchor_kind=anchor_kind,
                                   agg_kind=agg_kind).to(dev).eval()
            m_a.load_state_dict(model.state_dict()); m_a.use_anchor = False
            m_s = FusionWarpStereo(img_h=H, img_w=W, anchor_kind=anchor_kind,
                                   agg_kind=agg_kind).to(dev).eval()
            m_s.load_state_dict(model.state_dict()); m_s.delta_decoder.sparse = False
            d_anchor = (m_a(left, right)['preds'][-1] - base).abs().mean().item()
            d_sparse = (m_s(left, right)['preds'][-1] - base).abs().mean().item()
        print(f"[消融] 训练后输出平均差  no_anchor={d_anchor:.3f}  no_sparse={d_sparse:.3f}  "
              f"(>0 表示模块生效)")


if __name__ == '__main__':
    main()
