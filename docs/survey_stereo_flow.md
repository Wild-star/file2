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

## 4. 2026 最新进展与本设计的对照（含 prior-work 撞车声明）

> ⚠️ **诚实更正（2026-09）**：本设计核心 insight（相关体与 warp 互补融合）与
> **WAVE-Stereo**（arXiv:2607.13674，2026-07 公开）**高度撞车**。WAVE-Stereo 是本设计的
> **prior work**（公开时间早于本设计成文），而非「印证本设计」。本设计的核心 novelty
> 已被 WAVE-Stereo 抢先，须将其列为 prior work 引用并重新定位剩余差异化贡献。
> 逐点对照与定位见 `paper_design_fusionwarp.md` §9。

| 本设计模块 | 2026 工作 | 关系 |
|---|---|---|
| **核心融合（相关 + warp 互补）** | **WAVE-Stereo**（2607.13674，2026-07） | **prior work，核心思想撞车**（GeoWarp Correspondence Encoder 统一 matching search + residual alignment） |
| `SparseCorrAnchor`（窄带相关锚） | URS-Stereo（2607.06779，2026-07） | 并行方向：不确定性引导残差搜索，自适应重定位窄带中心 |
| `GEVCostAnchor`（轻量代价体） | LiteMatch（2606.31636，2026-06） | 并行方向：无 3D 卷积的代价体稳定化 + CVC-Loss |
| `GlobalMatcher`（全局注意力初始化） | LinStereo（2606.25437，2026-06） | 并行方向：线性复杂度全局注意力 + 深度先验热启动 |
| `TokenSparseViT`（token 稀疏） | WHTMix（2607.25234，2026-07） | 并行方向：Walsh-Hadamard 谱域全局 token 混合 |

**结论**：本设计落入一个 2026 年仍在活跃推进的主线——「把显式匹配与 warp 残差迭代互补地融合、并控制全局注意力成本」。但该主线的核心 novelty 已被 WAVE-Stereo 抢先发表，本设计只能以「WAVE-Stereo 的变体 / 差异化消融」定位，不能主张「统一相关与 warp」这一原创贡献。

---

## 5. 光流估计最新进展与可迁移灵感（2025-2026）

进一步检索光流估计核心方法（排除 SLAM/生成/事件/医学等外围应用），提炼可迁移到
「相关 + warp 互补」框架的灵感。**每条均标注创新性评估**（延续 §4 的诚实原则）。

### 5.1 核心方法清单

| 论文 | 会议 | 核心思想 |
|---|---|---|
| **FlowIt**（2603.28759） | arXiv 2026-03 | 全局匹配用**最优传输（OT/Sinkhorn）**做初始化，显式输出**遮挡 + 置信度**图，置信度引导细化（高置信→低置信传播） |
| **U²Flow**（2604.10056） | CVPR 2026 Oral | 无监督 + **联合估计光流与逐像素不确定性**（Laplace 最大似然），不确定性**引导自适应细化 + 调制平滑损失 + 双向融合** |
| **Removing Cost Volumes**（2510.13317） | ICCV 2025 | 代价体在**训练充分后失去重要性**；训练策略使代价体可被**蒸馏并在推理时移除**（提速 1.2×、内存 6×↓） |
| **FlowPainter**（2607.10140） | arXiv 2026-07 | 轻量网络预测粗略 flow + **置信度 mask**，仅对不确定硬区域做扩散细化（置信度引导的分区域处理） |
| **FreeFlow**（2609.11486） | ECCV 2026 | **去掉所有光流专用偏置**（相关体/warp/迭代），单一前馈 window/shifted/global 三注意力，SOTA |
| **Rethinking…Test-Time Scaling**（2605.08000） | CVPR 2026 ViSCALE | 冻结 DINO-v2 + 单目深度先验，**单次前向全局匹配，去掉迭代细化** |
| **QuantaFlow**（2608.00499） | arXiv 2026-08 | **表示构造嵌入迭代细化**：当前 flow 对齐→构造表示→驱动 warp 更新→指导下一轮表示 |
| **Finetuning Video Transformers**（2512.18684） | AAAI 2026 | 视频基础模型微调 + 线性解码器 + 迭代细化（骨干迁移） |
| **On Real-World Generalisability**（2607.10470） | ECCV 2026 | 诊断：大运动/光照最能预测真实精度，且大运动改进会**牺牲小运动** |

### 5.2 可迁移灵感与创新性评估

| 灵感 | 来源 | 对本设计的融合点 | 创新性评估 |
|---|---|---|---|
| **A. OT 全局匹配 + 置信度/遮挡显式化** | FlowIt | 把 `GlobalMatcher` 的 soft-argmax 换成 **Sinkhorn OT**；置信度/遮挡图复用为：① 引导迭代细化 ② 替代 saliency gate 做 token 稀疏 ③ 损失权重 | ⚠️ 光流侧 FlowIt 已做；**立体侧 OT 全局匹配已被 STTR（ICCV 2021）覆盖**；仅「OT 初始化 + 置信度引导 嵌入 相关+warp 迭代」这一特定组合可能是空白（增量级，非新范式） |
| **B. 不确定性引导自适应细化** | U²Flow / FlowPainter / LC-Flow | 显式逐像素不确定性 → ① 自适应窄带锚搜索半径 R（呼应 URS-Stereo）② 替代/增强 saliency gate ③ 把固定 MixtureLaplace 系数 pi 改为逐像素 | ❌ 已被 URS-Stereo（立体）+ U²Flow（光流）覆盖，创新性弱；仅可作消融 |
| **C. 代价体「训练蒸馏、推理移除」** | Removing Cost Volumes (ICCV25) | `GEVCostAnchor` 训练时用、推理时剪枝 → 进一步提速 | ⚠️ 光流侧已做；**立体窄带代价体的训练蒸馏移除，待查**（潜在差异化点） |
| **D. 无偏置前馈的对照** | FreeFlow / Rethinking | 作为论文的**讨论/反事实对照**：我们融合偏置，但 FreeFlow 证明无偏置也能 SOTA → 诚实讨论偏置的边际价值 | 非模块，是讨论点（增强论文诚实度） |

### 5.3 结论（已查证）

- **A（OT + 置信度/遮挡）**：立体侧 OT 全局匹配已被 **STTR（ICCV 2021）** 覆盖，创新性弱；
  仅「OT 初始化 + 置信度引导」嵌入「相关 + warp 互补」迭代这一**组合**可能是空白（增量级）。
- **C（代价体训练蒸馏、推理移除）**：光流侧 ICCV 2025 已做；双目立体侧针对性检索
  （`stereo + cost volume + distillation`，4 条结果均为 MVS/热成像，无直接覆盖）→
  **相对最空白、最值得做**的增量方向。
- **B（不确定性引导）**：已被 URS-Stereo（立体）+ U²Flow（光流）覆盖，仅可作消融。
- **D（无偏置前馈对照）**：讨论点，增强论文诚实度。

**结论**：唯一称得上「相对空白」的是 **C**——把 `GEVCostAnchor` 改成「训练时作为老师信号、
推理时蒸馏移除」的窄带代价体，与 Removing Cost Volumes（ICCV 2025）在光流侧的发现对齐、
但落到双目「相关+warp」框架。其余方向要么被覆盖、要么只是讨论点。
下一步建议聚焦 C，先做 POC 验证「代价体在训练后期是否确实失去重要性」。
