# Related Work：模块迁移依据（方法章节每一处缝合的论文支撑）

> 本文档为 `paper_design_fusionwarp.md` §2/§3 的 8 个迁移模块逐一提供**迁移依据**：
> 来源（完整引用）→ 迁移依据（为什么该机制适合 WAFT 的短板）→ 佐证链（该机制在
> 文献中被反复验证的证据）→ 缝合位置论证。所有有效性以充分训练为准，此处只论证
> **设计层面的合理性**，不作有效性判定。

---

## A1. GlobalMatcher —— 全局匹配初始化

**来源**
- GMFlow: "GMFlow: Learning Optical Flow via Global Matching" (CVPR 2022, arXiv:2111.13680) —— transformer 全局匹配 + soft-argmax 直接回归。
- SEA-RAFT: "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow" (ECCV 2024, arXiv:2405.14793) —— direct 初始回归替代零初始化。

**迁移依据**：WAFT 的 `prop_bins` 在**局部感受野**内做视差 bins 软分类，缺乏跨区域定位，
大视差/低纹理下易错。GMFlow 证明「全局匹配 + soft-argmax 直接回归」在少迭代下即可准确定位；
SEA-RAFT 证明「direct 初始回归」替代 RAFT 的零初始化能加速收敛并提升精度。两者共同支撑
「用全局匹配做初始视差」这一缝合。

**佐证链**：STTR (ICCV 2021, 2011.02910) 的全局匹配 + uniqueness 约束；FlowIt (2026,
2603.28759) 的 OT 全局匹配。全局匹配作为初始化在光流/立体是成立的主线。

**缝合**：替换 `prop_proj/prop_decoder/prop_bins_head`，输出 `d_gm`（初始视差）+ `g_feat`（全局上下文）。

---

## A2. CostAnchor —— 窄带代价体锚

**来源**
- RAFT-Stereo: "RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching" (3DV 2021, arXiv:2109.07547) —— 迭代内相关体检索。
- IGEV-Stereo: "Iterative Geometry Encoding Volume for Stereo Matching" (CVPR 2023, arXiv:2303.06615) —— combined geometry encoding volume。

**迁移依据**：WAFT 的 `warped_fmap2 = warp(f2, disp)` 是**当前视差下的单点对齐**，无多候选
匹配证据。RAFT-Stereo 证明「逐轮在当前视差附近检索相关体」提供匹配证据；IGEV 证明「几何编码体」
比裸相关更稳。这正是 WAFT 单点 warp 所缺的。

**佐证链**：GwcNet (CVPR 2019, 1903.04025) group-wise 相关；CREStereo (CVPR 2022,
2203.11483) 递归代价体。代价体作为匹配证据是成熟主线。

**缝合**：在匹配特征 `m1,m2` 上、当前 `disp` 的 `±R` 窄带做 group-wise 相关，零初始化投影得锚。
**关键**：锚不直接 concat，而是经 B1 门控融合——见「候选 A 失败」的教训。

---

## A3. 解耦代价体聚合 —— 空间×视差 2D 解耦

**来源**
- DBStereo: "Decoupling Bidirectional Geometric Representations of 4D cost volume with 2D convolution" (arXiv 2025, 2509.02415) —— 4D 代价体解耦为空间维 + 视差维，纯 2D 卷积聚合。

**迁移依据**：WAFT 是无 3D 卷积的轻量定位，代价体 3D 卷积会破坏效率。DBStereo 证明
「空间维与视差维分离、纯 2D 卷积」能以更低算力**超越**迭代法 IGEV 的精度。

**佐证链**：LiteMatch (2026, 2606.31636) 无 3D 卷积代价体稳定化；MAFNet (2025, 2512.04358)
频域分解 + 纯 2D 聚合。「轻量/无 3D 代价体」是 2025–2026 主线。

**缝合**：A2 窄带代价体的聚合层，用空间 2D 卷积 + 视差方向 1D 卷积解耦。

---

## B1. GatedFusion —— 空间门控融合

**来源**
- ACVNet: "Attention Concatenation Volume for Accurate and Efficient Stereo Matching" (CVPR 2022, arXiv:2203.02146) —— attention 融合多尺度代价体。
- CREStereo: "Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation" (CVPR 2022, arXiv:2203.11483)。

**迁移依据**：局部锚（纹理处可靠）与全局上下文（无纹理处可靠）的**可靠域互补**，需空间自适应
选择。ACVNet 证明「注意力门控融合」多尺度证据有效；Selective-Stereo (CVPR 2024, 2403.00486)
的 CSA 用注意力图作融合权重，进一步证明「用注意力做多路融合」在迭代立体匹配中是 SOTA 主线。

**佐证链**：ACVNet / Selective-Stereo / CREStereo 三者都证明「注意力/门控融合异构代价体」有效。

**缝合**：`g = σ(Conv([anchor, g_feat]))`，`fused = g·anchor + (1-g)·g_feat`。是「候选 A 简单
concat 失败」的修正——从无选择拼接升级为可学习门控。

---

## B2. Uncertainty —— 逐像素不确定性引导

**来源**
- U²Flow: "U²Flow: Unsupervised Uncertainty-aware Optical Flow" (CVPR 2026 Oral, arXiv:2604.10056) —— 联合估计 flow + aleatoric 不确定性，引导细化 + 调制损失。
- URS-Stereo: "Uncertainty-Guided Residual Search for Stereo Matching" (arXiv 2026, 2607.06779) —— 不确定性引导窄带残差搜索。

**迁移依据**：遮挡/无纹理是难例，固定损失/固定搜索半径无法自适应。U²Flow 证明「逐像素
不确定性引导细化 + 调制平滑损失」；URS-Stereo 证明「不确定性引导自适应重定位窄带中心」。
二者共同支撑「用 σ 统一引导门控、搜索半径、损失」。

**佐证链**：Kendall & Gal (NeurIPS 2017) 的 aleatoric uncertainty 经典；SEA-RAFT (ECCV 2024)
的 mixture-Laplace 隐含不确定性。不确定性引导是 2024–2026 主线。

**缝合**：迭代隐状态上 `σ = softplus(conv(net))`，复用三处：① 调制 B1 门控 ② 自适应 A2 半径 ③ 损失权重。

---

## B3. GlobalContext —— 全局上下文注意力

**来源**
- GREAT-Stereo: "Global Regulation and Excitation via Attention Tuning for Stereo Matching" (ICCV 2025, arXiv:2509.15891) —— SA(空间) + MA(epipolar) + VA(体积) 三注意力，即插即用注入迭代。

**迁移依据**：WAFT 迭代无全局上下文传播，无纹理/重复纹理区域无法从远处可靠区「借用」信息。
GREAT 证明「把全局上下文经注意力注入 RAFT/IGEV 迭代」显著提升病态区域精度（即插即用、
多项 leaderboard 第一）。这正是 WAFT 所缺的全局传播。

**佐证链**：LinStereo (2026, 2606.25437) 位置感知线性全局注意力；WHTMix (2026, 2607.25234)
谱域全局 token 混合。全局上下文注入是 2025–2026 主线。

**缝合**：迭代内对隐状态做 epipolar + 空间方向的轻量注意力，得全局上下文，与 `g_feat` 合并供 B1。

---

## C1. TokenSparseViT —— token 级选择性更新

**来源**
- DynamicViT: "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification" (NeurIPS 2021, arXiv:2106.02034) —— 轻量模块估计 token 重要性，注意力掩码动态剪枝冗余 token。

> ⚠️ 来源更正：此前误标 Selective-Stereo；其核心实为「多频率自适应融合（SRU+CSA）」，已移至
> B1 佐证链。token 稀疏的真实来源是 DynamicViT。

**迁移依据**：WAFT 的 `VitIter` 对全量 token 做注意力，但逐轮视差增量高度稀疏（本仓库 P0
诊断），与 DynamicViT「注意力本质稀疏、可剪 66% token 仅降 0.5% 精度、减 31–37% FLOPs」的
观察一致。故用 saliency 门控做 token 级选择性更新。

**佐证链**：DynamicViT (NeurIPS 2021) 为 token 剪枝开山；后续 EViT/ATS 等延续。稀疏更新是
成熟效率手段。

**缝合**：替代 `VitIter` 全量 token；`gate = σ(saliency)`，`h' = tok + gate⊙(Attn(tok)-tok)`。

---

## C2. 代价体训练蒸馏 —— 训练充分、推理移除

**来源**
- Removing Cost Volumes from Optical Flow Estimators (ICCV 2025, arXiv:2510.13317) —— 训练策略使代价体在训练后重要性下降，可蒸馏并在推理时移除（提速 1.2×、省 6×内存）。

**迁移依据**：A2 代价体在推理有开销，而该文证明代价体在训练充分后趋于冗余、可被蒸馏移除。
这支撑「A2 训练时作为完整锚 + 辅助监督 `d_cv`，训练后期蒸馏、推理移除」的缝合，与 WAFT
轻量定位一致。

**佐证链**：该方向较新（ICCV 2025 单一实证），佐证较少——如实标注，需充分训练复验。

**缝合**：A2 的训练/推理分离；训练时完整锚 + 辅助监督，推理时移除 A2 分支。

---

## 附：完整引用清单（按首次出现）

| 模块 | 论文 | 会议/年份 | arXiv |
|---|---|---|---|
| A1 | GMFlow | CVPR 2022 | 2111.13680 |
| A1 | SEA-RAFT | ECCV 2024 | 2405.14793 |
| A1 | STTR | ICCV 2021 | 2011.02910 |
| A1 | FlowIt | 2026 | 2603.28759 |
| A2 | RAFT-Stereo | 3DV 2021 | 2109.07547 |
| A2 | IGEV-Stereo | CVPR 2023 | 2303.06615 |
| A2 | GwcNet | CVPR 2019 | 1903.04025 |
| A2 | CREStereo | CVPR 2022 | 2203.11483 |
| A3 | DBStereo | 2025 | 2509.02415 |
| A3 | LiteMatch | 2026 | 2606.31636 |
| A3 | MAFNet | 2025 | 2512.04358 |
| B1 | ACVNet | CVPR 2022 | 2203.02146 |
| B1 | Selective-Stereo | CVPR 2024 | 2403.00486 |
| B2 | U²Flow | CVPR 2026 Oral | 2604.10056 |
| B2 | URS-Stereo | 2026 | 2607.06779 |
| B2 | Kendall & Gal | NeurIPS 2017 | — |
| B3 | GREAT-Stereo | ICCV 2025 | 2509.15891 |
| B3 | LinStereo | 2026 | 2606.25437 |
| B3 | WHTMix | 2026 | 2607.25234 |
| C1 | DynamicViT | NeurIPS 2021 | 2106.02034 |
| C2 | Removing Cost Volumes | ICCV 2025 | 2510.13317 |
