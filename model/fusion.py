# -*- coding: utf-8 -*-
"""
FusionWarp-Stereo 的融合模块（缝合到 WAFT-Stereo 基线）。

本文件提供三个可消融的模块，全部遵循「零初始化 = 手术安全」原则：
在 FUSION.ENABLED=True 但 USE_GLOBAL_INIT=False 时，前向输出与原始 WAFT 比特级一致
（anchor 输出恒为 0），保证加载预训练 WAFT 权重后行为等价原版，可安全微调。

  A. MatchingBranch   —— 从【原始图像】提取「匹配友好」特征（Step-1 结论：VFM 特征不是匹配代价）
  B. SparseCorrAnchor —— 在 disp±R 内做 group-wise【无聚合】相关，零初始化注入迭代
  B'. GEVCostAnchor   —— 可分离 3D 聚合的窄带代价体（与 B 同一插槽，作「无聚合 vs 聚合」对抗性消融）
  C. GlobalMatcher    —— 1/8 粗尺度交叉注意力 + 全范围相关 soft-argmax，直接回归初始视差

设计决策（与 WAVE-Stereo 的实质差异，详见 docs/paper_design_fusionwarp.md §9 修订）：
  - WAVE-Stereo 用 ConvGRU + PGCP(周期性补全局) + 几何编码体积(带 2D/3D 聚合)；
  - 本方案用 ViT(原生全局注意力) + 无聚合逐点相关锚 + VFM 先验 + 最小匹配分支。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import disp_warp


# --------------------------------------------------------------------------- #
# 0. 基础算子
# --------------------------------------------------------------------------- #
def group_corr(m1, m2, G):
    """group-wise 相关：通道分 G 组，组内归一化点积。返回 (B, G, h, w)。"""
    B, C, h, w = m1.shape
    Cg = C // G
    a = F.normalize(m1.view(B, G, Cg, h, w), dim=2)
    b = F.normalize(m2.view(B, G, Cg, h, w), dim=2)
    return (a * b).sum(dim=2)


# --------------------------------------------------------------------------- #
# A. MatchingBranch —— 匹配友好特征（独立于 VFM 编码器）
# --------------------------------------------------------------------------- #
class MatchingBranch(nn.Module):
    """从原始图像(归一化后)提取匹配友好特征，stride=4 → 1/4 分辨率。

    Step-1 实测：34K 参数的匹配分支即可把「GT 对齐后 argmax 命中率」从 46% 提到 87%，
    说明匹配信号可以廉价造出来，不必依赖 VFM 特征。
    """

    def __init__(self, ch=32, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 5, stride=2, padding=2), nn.InstanceNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.InstanceNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.InstanceNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch),
        )  # 两个 stride-2 → 1/4 分辨率

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# B. SparseCorrAnchor —— 无聚合逐点相关锚（核心注入）
# --------------------------------------------------------------------------- #
class SparseCorrAnchor(nn.Module):
    """在 disp±R 内做 group-wise 无聚合相关，零初始化聚合成 anchor 特征。

    与 WAVE-Stereo 的几何编码体积(带 2D/3D 聚合)不同：这里不做任何视差维/空间维聚合，
    只保留逐点相关分布，经零初始化 1×1/3×3 卷积映射到迭代隐空间。

    升级2（multi_scale=True）：在 1/4 和 1/8 两个尺度各做一条无聚合窄带相关，
    上采样对齐后 concat，提供多尺度匹配证据（呼应 LinStereo 的 HSCV，但仍无聚合）。
    """

    def __init__(self, C, G=8, R=4, out_ch=48, multi_scale=False):
        super().__init__()
        assert C % G == 0, f"MatchingBranch 通道 {C} 必须能被分组数 {G} 整除"
        self.G, self.R, self.multi_scale = G, R, multi_scale
        if multi_scale:
            half = out_ch // 2
            self.head1 = nn.Conv2d((2 * R + 1) * G, half, 3, padding=1)
            self.head2 = nn.Conv2d((2 * R + 1) * G, out_ch - half, 3, padding=1)
            for h in (self.head1, self.head2):
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)
        else:
            self.head = nn.Conv2d((2 * R + 1) * G, out_ch, 3, padding=1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _corr(self, m1, m2, disp, head):
        sims = []
        for o in range(-self.R, self.R + 1):
            w = disp_warp(m2, disp + o, padding_mode='zeros')
            sims.append(group_corr(m1, w, self.G))
        cost = torch.cat(sims, dim=1)  # (B, (2R+1)*G, h, w)
        return head(cost)

    def forward(self, m1, m2, disp):
        """m1/m2: (B, C, h, w) 匹配特征(1/4)；disp: (B, 1, h, w) 同尺度视差。"""
        if not self.multi_scale:
            return self._corr(m1, m2, disp, self.head)
        # 尺度 1：1/4
        f1 = self._corr(m1, m2, disp, self.head1)
        # 尺度 2：1/8（下采样 2×，视差 ×0.5）
        m1_s = F.avg_pool2d(m1, 2, 2)
        m2_s = F.avg_pool2d(m2, 2, 2)
        disp_s = F.avg_pool2d(disp, 2, 2) * 0.5
        f2 = self._corr(m1_s, m2_s, disp_s, self.head2)
        f2 = F.interpolate(f2, size=m1.shape[-2:], mode='bilinear', align_corners=True)
        return torch.cat([f1, f2], dim=1)


# --------------------------------------------------------------------------- #
# B'. GEVCostAnchor —— 可分离 3D 聚合的窄带代价体（对抗性消融对照组）
# --------------------------------------------------------------------------- #
class GEVCostAnchor(nn.Module):
    """轻量引用传统代价体：窄带「组合几何编码体」+ 2 层轻量 3D 聚合。

    与 SparseCorrAnchor 的区别：前者无聚合，这里是 3D 卷积正则化聚合。
    用于回答「3D 聚合是否带来增益」这一与 WAVE-Stereo 直接对立的命题
    （Step-3 实测：corr(无聚合) > gev(带聚合)）。
    """

    def __init__(self, C=32, G=4, K=9, R=8, Cv=8, out_ch=48, agg_kind='sep3d'):
        super().__init__()
        assert C % G == 0
        self.G, self.K, self.R = G, K, R
        self.agg_kind = agg_kind
        # 非均匀采样偏移：中心密、边缘疏
        t = torch.linspace(-1, 1, K)
        offs = (R * torch.sign(t) * t.abs() ** 2).tolist()
        offs[0], offs[-1] = -float(R), float(R)
        self.offsets = offs
        self.agg = self._build_agg(G + 1, Cv)
        self.cost_head = nn.Conv3d(Cv, 1, (1, 1, 1))
        self.feat_proj = nn.Conv2d(Cv, out_ch, 3, padding=1)
        # 零初始化输出 → 手术安全：feat=0，d_cv 退化为当前视差
        for m in (self.cost_head, self.feat_proj):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _sep_block(self, cin, cout):
        return nn.Sequential(
            nn.Conv3d(cin, cout, (1, 3, 3), padding=(0, 1, 1)), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0)), nn.ReLU(inplace=True))

    def _build_agg(self, cin, Cv):
        if self.agg_kind == 'sep3d':
            return nn.Sequential(self._sep_block(cin, Cv), self._sep_block(Cv, Cv))
        return nn.Sequential(  # full3d
            nn.Conv3d(cin, Cv, (3, 3, 3), padding=(1, 1, 1)), nn.ReLU(inplace=True),
            nn.Conv3d(Cv, Cv, (3, 3, 3), padding=(1, 1, 1)), nn.ReLU(inplace=True))

    def forward(self, m1, m2, disp):
        B, C, h, w = m1.shape
        Cg = C // self.G
        corrs, geos = [], []
        for o in self.offsets:
            d = disp + o
            w2 = disp_warp(m2, d, padding_mode='zeros')
            g = (F.normalize(m1.view(B, self.G, Cg, h, w), dim=2) *
                 F.normalize(w2.view(B, self.G, Cg, h, w), dim=2)).sum(dim=2)  # (B,G,h,w)
            corrs.append(g)
            geos.append(d)
        vol = torch.cat([torch.stack(corrs, dim=2), torch.stack(geos, dim=2)], dim=1)  # (B,G+1,K,h,w)
        agg = self.agg(vol)                                     # (B,Cv,K,h,w)
        cost = self.cost_head(agg).squeeze(1)                   # (B,K,h,w)
        prob = F.softmax(cost, dim=1)
        dk = torch.stack([disp + o for o in self.offsets], dim=1).squeeze(2)  # (B,K,h,w)
        d_cv = (prob * dk).sum(dim=1, keepdim=True)             # (B,1,h,w)
        feat = (prob.unsqueeze(1) * agg).sum(dim=2)             # (B,Cv,h,w)
        feat = self.feat_proj(feat)                             # (B,out_ch,h,w)
        return d_cv, feat


# --------------------------------------------------------------------------- #
# C. GlobalMatcher —— 全局匹配初始视差（直接回归）
# --------------------------------------------------------------------------- #
def _mha(q, k, v, heads):
    B, N, C = q.shape
    d = C // heads
    q = q.view(B, N, heads, d).transpose(1, 2)
    k = k.view(B, N, heads, d).transpose(1, 2)
    v = v.view(B, N, heads, d).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(B, N, C)


class CrossAttentionLayer(nn.Module):
    """左↔右 token 交叉注意力（增强特征，缓解低纹理/遮挡歧义）。

    带可学习 Q/K/V 投影（左右共享权重），让注意力真正可学习「该关注什么」。
    注意：不显式加位置编码——输入 fmap 来自 DAv2/DINOv3 的 ViT 编码器，
    其 token 已隐含位置信息（这与从零训练的小编码器不同）。
    """

    def __init__(self, C, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.heads = heads
        # 可学习 Q/K/V 投影（左右共享）
        self.to_q = nn.Linear(C, C)
        self.to_k = nn.Linear(C, C)
        self.to_v = nn.Linear(C, C)
        self.ff = nn.Sequential(nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, left, right):
        l = self.norm1(left)
        r = self.norm1(right)
        l2 = _mha(self.to_q(l), self.to_k(r), self.to_v(r), self.heads) + left
        l2 = self.ff(self.norm2(l2)) + l2
        r2 = _mha(self.to_q(r), self.to_k(l), self.to_v(l), self.heads) + right
        r2 = self.ff(self.norm2(r2)) + r2
        return l2, r2


class GlobalMatcher(nn.Module):
    """1/8 粗尺度：交叉注意力 + 全范围相关 soft-argmax，直接回归初始视差 d_gm。

    同时输出全局上下文特征 g_feat（1/2 尺度），供 GatedFusion 门控融合使用。
    d_gm 上采样回 1/2 尺度（视差 ×4），作为迭代的初始视差（可开关 USE_GLOBAL_INIT）。
    """

    def __init__(self, C, heads=4, n_disp=100):
        super().__init__()
        assert C % heads == 0, f"通道 {C} 必须能被 head 数 {heads} 整除"
        self.C = C
        self.heads = heads
        self.n_disp = n_disp
        self.cross = CrossAttentionLayer(C, heads)
        self.gfeat_proj = nn.Conv2d(2 * C, C, 1)   # 全局上下文特征投影（cat[l,r] → C）
        self.disp_idx = torch.arange(n_disp).view(1, n_disp, 1, 1).float()

    def forward(self, f1, f2):
        B = f1.shape[0]
        # 1/2 → 1/8
        f1c = F.avg_pool2d(f1, 4, 4)
        f2c = F.avg_pool2d(f2, 4, 4)
        h, w = f1c.shape[-2:]
        # tokens + 交叉注意力
        l = f1c.flatten(2).transpose(1, 2)  # (B, N, C)
        r = f2c.flatten(2).transpose(1, 2)
        l, r = self.cross(l, r)
        l = l.transpose(1, 2).reshape(B, self.C, h, w)
        r = r.transpose(1, 2).reshape(B, self.C, h, w)
        # 全局上下文特征 g_feat（1/8 → 1/2 尺度）
        g_feat = F.interpolate(torch.cat([l, r], dim=1), scale_factor=4,
                               mode='bilinear', align_corners=True)  # (B, 2C, H/2, W/2)
        g_feat = self.gfeat_proj(g_feat)                              # (B, C, H/2, W/2)
        # 全范围相关 + soft-argmax（1/8 尺度）
        ln = F.normalize(l, dim=1)
        cost = []
        for d in range(self.n_disp):
            w_r = disp_warp(
                r, torch.full((B, 1, h, w), float(d), device=f1.device, dtype=f1.dtype),
                padding_mode='zeros')
            cost.append((ln * F.normalize(w_r, dim=1)).sum(1, keepdim=True))
        cost = torch.cat(cost, dim=1)                       # (B, n_disp, h, w)
        prob = F.softmax(cost, dim=1)
        d_gm = torch.sum(prob * self.disp_idx.to(f1.device), dim=1, keepdim=True)  # (B,1,h,w) @1/8
        # 上采样回 1/2 尺度：空间 ×4，视差 ×4
        d_gm = F.interpolate(d_gm, scale_factor=4, mode='bilinear', align_corners=True) * 4.0
        return d_gm, g_feat  # (B,1,H/2,W/2), (B,C,H/2,W/2)


class GatedFusion(nn.Module):
    """可学习门控融合：空间自适应地融合「局部相关锚」与「全局上下文」。

    gate = σ(conv([anchor, g_feat]))，fused = gate·anchor + (1-gate)·g_feat。
    输出乘零初始化的 out_scale → 初始 fused=0（手术安全，等价原版），训练时学起来。
    """

    def __init__(self, C):
        super().__init__()
        self.gate = nn.Conv2d(2 * C, 1, 3, padding=1)
        self.out_scale = nn.Parameter(torch.zeros(1))   # 零初始化 → fused=0

    def forward(self, anchor, g_feat):
        g = torch.sigmoid(self.gate(torch.cat([anchor, g_feat], dim=1)))
        fused = g * anchor + (1 - g) * g_feat
        return fused * self.out_scale


# --------------------------------------------------------------------------- #
# D. GWCEFusion —— 补充式 GWCE 三路并行编码旁路（学习 WAVE GWCE + PCW warping volume）
# --------------------------------------------------------------------------- #
class GWCEFusion(nn.Module):
    """在 WAFT 原始 delta 迭代之外，新增的「补充式」三路并行编码旁路。

    学习 WAVE-Stereo 的 GWCE（三路独立编码 + Fusion）与 PCW-Net 的 warping volume
    （用当前视差把相关检索范围缩小到窄带）。三路互补线索：
      ① correlation    ：1/4 尺度窄带相关（disp±R）→ 1×1 + 3×3 → 上采样到 1/2
      ② warp 对齐      ：cat[fmap1, warped_fmap2] → 两个 3×3（跨视角对齐残差）
      ③ disparity prior：disp → 7×7 大核 + 3×3（当前几何状态的空间分布）
    三路 concat 后经零初始化 1×1 卷积 Fusion 到隐维度 → 作为残差加到 net。

    「补充而非替换」：WAFT 的 delta_proj/delta_decoder 完全不动，本模块输出仅作残差。
    Fusion 零初始化 → 输出初始恒 0，加载预训练 WAFT 后行为等价原版（手术安全），
    消融干净（关某分支 → 少一路 concat，Fusion 输入通道相应减少，仍零初始化）。
    """

    def __init__(self, C_match, C_enc, hidden, G=8, R=4, disp_kernel=7, fuse_ch=32,
                 use_corr=True, use_warp=True, use_disp_prior=True):
        super().__init__()
        assert any([use_corr, use_warp, use_disp_prior]), "GWCE 至少开一路分支"
        self.use_corr = use_corr
        self.use_warp = use_warp
        self.use_disp_prior = use_disp_prior
        self.G, self.R = G, R

        # ① correlation 分支（学 WAVE correlation branch：1×1 压缩 + 3×3 空间上下文）
        if use_corr:
            assert C_match % G == 0, f"匹配通道 {C_match} 必须能被分组数 {G} 整除"
            self.enc_c = nn.Sequential(
                nn.Conv2d((2 * R + 1) * G, fuse_ch, 1), nn.ReLU(inplace=True),
                nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1))

        # ② warp 对齐分支（学 WAVE cross-view warping branch：两个 3×3）
        if use_warp:
            self.enc_w = nn.Sequential(
                nn.Conv2d(2 * C_enc, fuse_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1))

        # ③ disparity prior 分支（学 WAVE disparity prior branch：7×7 大核 + 3×3）
        if use_disp_prior:
            self.enc_d = nn.Sequential(
                nn.Conv2d(1, fuse_ch, disp_kernel, padding=disp_kernel // 2), nn.ReLU(inplace=True),
                nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1))

        # Fusion：三路 concat → 隐维度，零初始化 → 输出 0（手术安全）
        n_branches = int(use_corr) + int(use_warp) + int(use_disp_prior)
        self.fusion = nn.Conv2d(n_branches * fuse_ch, hidden, 1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, m1, m2, fmap1, warped_fmap2, disp):
        """m1/m2: (B, C_match, H/4, W/4) 匹配特征；fmap1/warped_fmap2: (B, C_enc, H/2, W/2)；
        disp: (B, 1, H/2, W/2) 当前视差。返回 (B, hidden, H/2, W/2) 残差特征（初始 0）。"""
        parts = []
        if self.use_corr:
            # 1/2 → 1/4 视差（像素单位 ×0.5），窄带相关（PCW warping volume）
            disp_q = F.interpolate(disp, scale_factor=0.5, mode='bilinear', align_corners=True) * 0.5
            corrs = [group_corr(m1, disp_warp(m2, disp_q + o, padding_mode='zeros'), self.G)
                     for o in range(-self.R, self.R + 1)]
            cost = torch.cat(corrs, dim=1)                       # (B, (2R+1)*G, H/4, W/4)
            x_c = self.enc_c(cost)                               # (B, fuse_ch, H/4, W/4)
            x_c = F.interpolate(x_c, size=fmap1.shape[-2:], mode='bilinear', align_corners=True)
            parts.append(x_c)
        if self.use_warp:
            parts.append(self.enc_w(torch.cat([fmap1, warped_fmap2], dim=1)))
        if self.use_disp_prior:
            parts.append(self.enc_d(disp))
        return self.fusion(torch.cat(parts, dim=1))              # (B, hidden, H/2, W/2)
