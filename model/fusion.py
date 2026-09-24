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
    """

    def __init__(self, C, G=8, R=4, out_ch=48):
        super().__init__()
        assert C % G == 0, f"MatchingBranch 通道 {C} 必须能被分组数 {G} 整除"
        self.G, self.R = G, R
        self.head = nn.Conv2d((2 * R + 1) * G, out_ch, 3, padding=1)
        # 零初始化 → 前向恒为 0，等价原版
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, m1, m2, disp):
        """m1/m2: (B, C, h, w) 匹配特征(1/4)；disp: (B, 1, h, w) 同尺度视差。"""
        sims = []
        for o in range(-self.R, self.R + 1):
            w = disp_warp(m2, disp + o, padding_mode='zeros')
            sims.append(group_corr(m1, w, self.G))
        cost = torch.cat(sims, dim=1)  # (B, (2R+1)*G, h, w)
        return self.head(cost)


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

    注意：不显式加位置编码——输入 fmap 来自 DAv2/DINOv3 的 ViT 编码器，
    其 token 已隐含位置信息（这与从零训练的小编码器不同）。
    """

    def __init__(self, C, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.heads = heads
        self.ff = nn.Sequential(nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, left, right):
        l2 = _mha(self.norm1(left), self.norm1(right), self.norm1(right), self.heads) + left
        l2 = self.ff(self.norm2(l2)) + l2
        r2 = _mha(self.norm1(right), self.norm1(left), self.norm1(left), self.heads) + right
        r2 = self.ff(self.norm2(r2)) + r2
        return l2, r2


class GlobalMatcher(nn.Module):
    """1/8 粗尺度：交叉注意力 + 全范围相关 soft-argmax，直接回归初始视差 d_gm。

    替换 WAFT prop 分支的「bins 软分类 + soft-argmax」起点（可开关 USE_GLOBAL_INIT）。
    d_gm 上采样回 1/2 尺度（视差 ×4），作为迭代的初始视差。
    """

    def __init__(self, C, heads=4, n_disp=100):
        super().__init__()
        assert C % heads == 0, f"通道 {C} 必须能被 head 数 {heads} 整除"
        self.C = C
        self.heads = heads
        self.n_disp = n_disp
        self.cross = CrossAttentionLayer(C, heads)
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
        # 全范围相关 + soft-argmax（1/8 尺度）
        l = F.normalize(l, dim=1)
        cost = []
        for d in range(self.n_disp):
            w_r = disp_warp(
                r, torch.full((B, 1, h, w), float(d), device=f1.device, dtype=f1.dtype),
                padding_mode='zeros')
            cost.append((l * F.normalize(w_r, dim=1)).sum(1, keepdim=True))
        cost = torch.cat(cost, dim=1)                       # (B, n_disp, h, w)
        prob = F.softmax(cost, dim=1)
        d_gm = torch.sum(prob * self.disp_idx.to(f1.device), dim=1, keepdim=True)  # (B,1,h,w) @1/8
        # 上采样回 1/2 尺度：空间 ×4，视差 ×4
        d_gm = F.interpolate(d_gm, scale_factor=4, mode='bilinear', align_corners=True) * 4.0
        return d_gm  # (B, 1, H/2, W/2)
