# 立体匹配 / 光流估计 近年方法调研与模块拆解

> 目标：从近 5 年（2020–2025）优秀的双目立体匹配与光流估计论文中，提炼**可复用的模块级设计**，
> 作为本仓库 WAFT-Stereo 之上「新论文设计」的融合素材。每条都标注了「借鉴来源 → 我们怎么用」。

## 0. 本仓库的底座：WAFT-Stereo

- **WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching**（arXiv:2603.24836，Princeton 王奕涵 / 邓嘉）
- 核心主张：**不需要代价体（cost volume）**，仅靠逐轮 warping + 场变换（field transform）即可达到 ETH3D/KITTI/Middlebury 第一，且快 1.8–6.7×。
- 结构：特征编码器（DAv2/DINOv3，输出 context 特征 `fmap1/fmap2/net`）→ 传播解码器给出初始视差分布 → 多轮迭代：
  `warp fmap2` → `delta_proj(cat[fmap1, warped_fmap2, net, disp])` → ViT 解码器 → 回归 `delta_disp/mask` → convex upsample。
- **可改进点（本调研的切入点）**：warping 只做局部对齐，低纹理 / 大视差 / 遮挡处缺一个「全局匹配信号」；
  ViT 每轮在全部 token 上跑（本仓库 P0 诊断已量化：Δdisp 高度集中，存在 token 稀疏空间）。

---

## 1. 光流估计

### 1.1 RAFT — 迭代 + 相关查找 + convex upsampling
- **RAFT: Recurrent All-Pairs Field Transforms for Optical Flow**（ECCV 2020，arXiv:2003.12039，Teed & Deng）
- 模块：全对 4D 相关体；GRU 迭代更新；`convex upsampling`（9 邻域学习权重上采样，边缘锐利）。
- 借鉴：**迭代细化 + convex upsampling**（本仓库 WAFT 已继承 RAFT 系，属同源）。

### 1.2 GMFlow — 把匹配重构为「全局特征匹配」
- **GMFlow: Learning Optical Flow via Global Matching**（CVPR 2022 Oral，arXiv:2111.13680）
- 关键思想：抛弃 4D 代价体，用 **transformer 增强特征 + softmax 全局匹配** 直接求对应，一次匹配 + 一次细化即可超过 RAFT 31 次细化。
- 模块三件套：`customized Transformer 特征增强` → `correlation + softmax 全局匹配` → `self-attention 流传播`。
- 借鉴：**GlobalMatcher**——在粗尺度（1/8）对左右特征做交叉注意力增强，再做**全视差范围相关 + soft-argmax**，直接回归初始视差（同时天然处理大视差/低纹理）。

### 1.3 GMFlowNet — 全局匹配 + 残差 4D 细化
- **Global Matching with Overlapping Attention for Optical Flow Estimation**（CVPR 2022，arXiv:2203.11335）
- 在全局匹配后再接一个残差 4D 代价体做局部细化；提出 patch-based overlapping attention 提取大上下文特征。
- 借鉴：**「全局粗匹配 + 局部细匹配」的两段式结构**，与我们的「全局初始视差 → 迭代局部 refine」直接对应。

### 1.4 FlowFormer / FlowFormer++ — 代价体 token 化 + transformer 编码
- **FlowFormer**（ECCV 2022，arXiv:2203.16194）：把 4D 代价体 token 化，用 alternate-group transformer (AGT) 编码成 cost memory，再用递归 transformer 解码器 + 动态位置查询解码。
- **FlowFormer++**（CVPR 2023，arXiv:2303.01237）：Masked Cost Volume Autoencoding (MCVA) 预训练。
- 借鉴：**transformer 作为代价/匹配信息的编码器**（我们的 GlobalMatcher 交叉注意力 + ViT 迭代解码器同属这一脉络）。

### 1.5 SEA-RAFT — 简单高效 + 鲁棒损失 + 直接初始回归
- **SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow**（ECCV 2024，arXiv:2405.14793，与 WAFT 同作者组）
- 三个改进：① **mixture of Laplace 损失**（比 L1 更鲁棒、梯度更稳）；② **直接回归初始 flow**（加速迭代收敛）；③ 刚体运动预训练（提升泛化）。
- 借鉴：**MixtureLaplaceLoss**（我们的训练目标）+ **d_gm 直接初始视差回归 + 辅助监督**（GlobalMatcher 的输出直接监督）。

---

## 2. 双目立体匹配

### 2.1 经典代价体路线（对照）
- **PSMNet**（CVPR 2018，arXiv:1803.08669）：SPP 上下文 + 3D 沙漏代价聚合。→ 代价体路线的代表。
- **GwcNet**（CVPR 2019，arXiv:1903.04025）：**group-wise correlation** 代价体（按通道分组相关，信息不坍缩、更省参）。
- 借鉴：**group-wise correlation**——我们把它用于窄带锚（SparseCorrAnchor），而非全尺寸代价体。

### 2.2 STTR — 序列到序列的 transformer 匹配（无代价体）
- **STTR**（ICCV 2021 Oral，arXiv:2011.02910）：用 position + attention 做 dense pixel matching，**释放固定视差范围限制**、显式给出遮挡与置信度、加 uniqueness 约束。
- 借鉴：**transformer 全局匹配 + 置信度**的思想（我们的 GlobalMatcher 不依赖固定窄带，全范围匹配）。

### 2.3 RAFT-Stereo — 多级 GRU + 相关查找
- **RAFT-Stereo**（CVPR 2022，arXiv:2109.07547）：把 RAFT 搬到双目，**multi-level ConvGRU** 跨尺度传播；在相关体上做**局部查找**。
- 借鉴：**在当前位置附近做稀疏相关查找**——即我们 SparseCorrAnchor 的「窄带 ±R 相关」，只查局部、内存友好。

### 2.4 CREStereo — 级联递归 + 自适应分组相关
- **Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation**（CVPR 2022，arXiv:2203.11483）
- 模块：**自适应 group correlation**（容忍非理想校正）；级联 + 递归 coarse-to-fine 细化。
- 借鉴：自适应/分组相关（我们的 group-wise 窄带锚）+ 级联细化。

### 2.5 ACVNet — 注意力拼接体（attention volume）
- **ACVNet**（CVPR 2022，arXiv:2203.02146）：从相关线索生成 **attention 权重**，抑制冗余、增强匹配信息；multi-level **adaptive patch matching** 提升无纹理区匹配判别力。
- 借鉴：**用注意力/门控融合相关信号**——即我们的 GatedFusion（可学习门控融合锚与全局上下文）。

### 2.6 IGEV-Stereo — 几何编码体 + 迭代索引
- **IGEV-Stereo**（CVPR 2023，arXiv:2303.06615）：**combined geometry encoding volume (GEV)** 同时编码几何、上下文与局部匹配细节，迭代索引更新视差；GEV 回归一个更准的**起始点**加速 ConvGRU 收敛。
- 借鉴：**「更准的起始点 + 迭代索引」**——我们的 d_gm 全局初始视差承担同样职责。

### 2.7 Selective-Stereo — 自适应多频选择（选择性更新单元）
- **Selective-Stereo**（CVPR 2024，arXiv:2403.00486）：迭代优化难以同时保留高频边缘与低频平滑（固定感受野），提出 **Selective Recurrent Unit (SRU)** + **Contextual Spatial Attention (CSA)** 自适应多频融合。
- 借鉴：**「选择性/自适应」地融合与更新**——与我们的 token 稀疏更新（哪些区域需要本轮更新）在动机上同源：不是所有位置都需要同等的迭代计算。

### 2.8 MoCha-Stereo — motif 通道注意力（边缘结构）
- **MoCha-Stereo**（CVPR 2024，arXiv:2404.06842）：Motif Channel Correlation Volume (MCCV) 捕捉几何结构 motif 通道，改善边缘细节；Reconstruction Error Motif Penalty (REMP) 细化全分辨率。
- 借鉴：通道级结构化注意力（可作为 GatedFusion 的通道门控变体方向）。

### 2.9 FoundationStereo — 零样本泛化 + 侧调骨干 + 长程上下文
- **FoundationStereo**（CVPR 2025，arXiv:2501.09898）：side-tuning 特征骨干（适配视觉基础模型的单目先验以缩小 sim-to-real 差距）；**long-range context reasoning** 做代价体滤波。
- 借鉴：**side-tuning / LoRA 骨干适配**（WAFT 已用 LoRA）+ 长程上下文注入代价/匹配（我们的 GlobalMatcher 全局上下文）。

---

## 3. 模块 → 融合映射表（本文设计）

| 借鉴模块 | 来源 | 在本文中的落点 |
|---|---|---|
| 全局特征匹配（transformer + softmax 相关） | GMFlow / STTR | `GlobalMatcher`：1/8 交叉注意力 + 全视差相关 → `d_gm` + 全局上下文 `g_feat` |
| 直接初始回归 + 辅助监督 | SEA-RAFT | `d_gm` 直接监督（1/2 尺度混合拉普拉斯） |
| group-wise 窄带相关（局部查找） | RAFT-Stereo / GwcNet / CREStereo | `SparseCorrAnchor`：disp±R 分组相关，零初始化 |
| **窄带代价体 + 3D 聚合 + 几何编码** | PSMNet / GwcNet / IGEV | `GEVCostAnchor`（`ANCHOR_KIND=gev`）：1/4 尺度组合几何编码体 + 轻量 3D 正则化 |
| 注意力/门控融合 | ACVNet / CREStereo | `GatedFusion`：σ(conv([anchor,g_feat])) 门控 |
| 选择性/稀疏迭代更新 | Selective-Stereo + 本仓库 P0 诊断 | `TokenSparseViT`：token 级 saliency 门控 |
| 鲁棒损失 | SEA-RAFT | `MixtureLaplaceLoss` |
| 迭代细化 + convex upsampling | RAFT / WAFT | 保持 WAFT 迭代 + convex upsampling |

---

## 4. 2026 最新进展与本设计的印证（截至 2026-09）

上文各模块的直接引用多为 2018–2024 年经典。经 arXiv 最新检索，2026 年多个工作**直接印证或强化**本设计的核心方向（尤其「相关体与 warp 互补」这一融合主线）：

| 本设计模块 | 2026 最新工作 | 印证/强化点 |
|---|---|---|
| **核心融合（相关 + warp 互补）** | **WAVE-Stereo**（arXiv:2607.13674，2026-07） | 明确指出「correlation volumes 与 feature warping 是互补匹配线索」，GeoWarp Correspondence Encoder 在 ConvGRU 输入处并行编码匹配搜索 + 残差对齐 + 视差先验 —— 与本设计「SparseCorrAnchor + warp 残差」完全同构 |
| `SparseCorrAnchor`（窄带相关锚） | **URS-Stereo**（arXiv:2607.06779，2026-07） | 不确定性引导残差搜索：预测传播视差的可靠性 + 残差搜索偏移，**自适应重定位局部代价体中心**（把固定 ±R 窄带升级为 uncertainty-aware） |
| `GEVCostAnchor`（轻量代价体） | **LiteMatch**（arXiv:2606.31636，2026-06） | 轻量零样本：**无 3D 卷积的代价体稳定化** + CVC-Loss（代价体一致性损失）—— 印证「轻量引用代价体」方向，且给出比「轻量 3D」更进一步的「无 3D」方案 |
| `GlobalMatcher`（全局注意力初始化） | **LinStereo**（arXiv:2606.25437，2026-06） | Position-Aware Linear Attention 以**线性复杂度**做全局聚合 + Depth Prior Initialization（深度先验热启动）—— 对应本设计 GlobalMatcher 的全局匹配与「直接初始回归」 |
| `TokenSparseViT`（token 稀疏） | **WHTMix**（arXiv:2607.25234，2026-07） | 用数据无关的 **Walsh-Hadamard 变换域全局 token 混合**（log-linear 成本）替代全局自注意力、保留左右 cross-attention —— 与本设计「降低每轮 token 全量计算」同目标，是更激进的替代方案 |

**结论**：本设计的模块组合并非「拼凑旧方法」，而是落在一个 2026 年仍在活跃推进的主线上——「把显式匹配（相关/代价体）与 warp 残差迭代互补地融合、并控制全局注意力的成本」。若要强化论文的时效性，上述 5 篇 2026 工作可直接作为本设计的最新文献支撑与改进方向（尤其 WAVE-Stereo 与 URS-Stereo 可作为 related work 的「并行/后续印证」）。
