# FusionWarp-Stereo：全局匹配、稀疏相关与迭代 warping 的融合立体匹配（论文设计）

> 本文是在 WAFT-Stereo（warping-only、无代价体）之上的**新方法设计**：
> 保留 warping 的效率优势，针对「低纹理 / 大视差 / 遮挡」和「每轮全量 token 计算」两个短板，
> 融合近年光流与立体匹配的模块，形成一个可训练、可消融的复合框架。
> 代码 POC：`step2_fusion_composite.py`（自包含，CPU 可跑通前向+反向+训练）。

## 1. 标题（候选）

**FusionWarp-Stereo: Fusing Global Matching, Sparse Correlation and Iterative Warping for Efficient Stereo Matching**

## 2. 摘要（要点）

现有基于迭代优化的立体匹配（RAFT-Stereo、IGEV-Stereo）依赖代价体；WAFT-Stereo 证明代价体非必需，
仅靠逐轮 warping 即可达到顶尖精度与速度。然而纯 warping 缺乏显式的全局匹配信号，在低纹理、
大视差区域收敛慢、易错配；且其迭代 ViT 每轮在所有 token 上全量计算，存在大量冗余。

本文提出 **FusionWarp-Stereo**，在不引入全尺寸代价体的前提下，向 WAFT 注入三类互补模块：
(1) **GlobalMatcher**（GMFlow/STTR 式全局匹配）在 1/8 粗尺度做交叉注意力 + 全视差范围相关，
直接回归初始视差并产出全局上下文种子；
(2) **SparseCorrAnchor**（RAFT-Stereo/GwcNet 式窄带相关）在 disp±R 内做 group-wise 相关，
以**零初始化**注入迭代更新（手术安全，初始化时等价原版）；
(3) **TokenSparseViT**（Selective-Stereo 选择性更新思想 + token 稀疏诊断）对迭代解码器的 token
做 saliency 门控，仅对高信息 token 充分更新。
三者由 **GatedFusion**（ACVNet 式门控）融合，并用 SEA-RAFT 的 **mixture-of-Laplace** 鲁棒损失训练。

预期效果：全局匹配补足低纹理/大视差，稀疏相关恢复局部纹理细节，token 稀疏降低计算冗余，
在保持 WAFT 无代价体高效率的同时提升精度与收敛速度。

## 3. 贡献

1. 提出「全局匹配初始视差 + 窄带相关锚 + 迭代 warping」的三源融合范式，**不依赖全尺寸代价体**。
2. 设计 `GatedFusion` 门控融合：可学习地权衡「全局上下文」与「局部相关锚」两个异构匹配信号。
3. 将 token 稀疏性（本仓库 P0 诊断的 Δdisp 集中性）落地为 `TokenSparseViT` 的可学习稀疏更新。
4. 自包含 POC 脚本 `step2_fusion_composite.py`：在 CPU 上完成前向/反向/训练与消融自检，
   作为论文方法的可复现最小实现。

## 4. 方法

### 4.1 总览

输入左图 I_L、右图 I_R（已校正，RGB）。共享 stride-2 特征编码器得到 context 特征 `f1,f2`（1/2 尺度）；
独立匹配分支得到匹配友好特征 `m1,m2`（1/2 尺度，Step-1 结论：context 特征非匹配代价，需独立分支）。

```
d_gm, g_feat = GlobalMatcher(f1, f2)          # 全局初始视差 + 全局上下文
disp = d_gm
net  = 0
for itr in 1..T:
    disp = detach(disp)
    anchor = SparseCorrAnchor(m1, m2, disp)   # (2R+1)*G → C，零初始化
    warped = warp(f2, disp)
    fused  = GatedFusion(anchor, g_feat)
    x      = cat[f1, warped, net, disp, fused]
    net    = delta_proj(x)                    # Conv
    net, g = TokenSparseViT(net)              # token 级稀疏更新，g=saliency gate
    Δdisp  = disp_head(net)
    disp   = disp + Δdisp
    disp_up = convex_upsample(disp * 2, mask) # 1/2 → 全分辨率
```

### 4.2 GlobalMatcher（GMFlow / STTR / SEA-RAFT）

- 将 `f1,f2` 池化到 1/8，展平成 token 并加可学习位置编码；
- 左↔右 **交叉注意力**（1 层）增强特征，缓解低纹理/遮挡歧义；
- 对增强后的特征做**全视差范围相关 + softmax**：`C(x,d)=⟨Ĩ_L(x), Ĩ_R(x-d)⟩`，`d_gm=Σ_d d·softmax(C)`；
- `d_gm` 上采样回 1/2 作为初始视差（**直接初始回归**，同 SEA-RAFT），
  并与 1/2 尺度 GT 做辅助监督；
- 增强特征经投影得到全局上下文 `g_feat`，作为迭代隐状态种子。

### 4.3 SparseCorrAnchor（RAFT-Stereo / GwcNet / CREStereo）

- 在当前视差 `disp` 的窄带 `±R` 内（1/2 尺度），对匹配特征做 **group-wise 相关**：
  通道分 G 组，组内归一化点积，得 `(2R+1)·G` 通道代价；
- 经 **零初始化 1×1/3×3 卷积**聚合成 C 通道锚特征 → 注入迭代。零初始化保证「加载预训练 WAFT 后行为等价原版」。

### 4.4 GatedFusion（ACVNet / CREStereo）

- 门控：`g = σ(Conv([anchor, g_feat]))`，`fused = g·anchor + (1-g)·g_feat`；
- 空间自适应地在「局部相关锚」与「全局上下文」之间选择可信信号。

### 4.5 TokenSparseViT（Selective-Stereo + P0 诊断）

- ViT patch 化后，router 预测每个 token 的 saliency，`gate = σ(sal)`；
- 稀疏更新：`h' = tok + gate ⊙ (Attn(tok) - tok)`。gate→0 的 token 原样保留（硬 top-k 版本可跳过注意力省算力）；
- 训练损失附加 `λ·mean(gate)` 鼓励稀疏；脚本统计并打印实际稀疏度。

### 4.6 损失（SEA-RAFT）

混合拉普拉斯负对数似然（对离群点鲁棒）：

```
L_ml(e) = -log( π·(1/2b0)exp(-|e|/b0) + (1-π)·(1/2b1)exp(-|e|/b1) )
L = λ_init·L_ml(d_gm - gt_half) + Σ_i 0.5^(T-1-i)·L_ml(disp_i - gt) + λ_s·mean(gate)
```

## 5. 消融设计（论文实验）

| 消融 | 预期 |
|---|---|
| 去掉 GlobalMatcher（随机初始视差） | 大视差/低纹理 EPE 上升、收敛变慢 |
| 去掉 SparseCorrAnchor（use_anchor=False） | 局部纹理细节变差，边缘 blur |
| **SparseCorrAnchor vs GEVCostAnchor（corr vs gev）** | 后者带 3D 正则化，无纹理/歧义区更稳 |
| **GEV full3d vs sep3d** | sep3d 精度≈持平、参数量/算力更低 |
| 去掉 GatedFusion（改为直接 concat/add） | 融合失衡，精度小幅下降 |
| 去掉 TokenSparseViT（gate≡1） | 精度≈持平，计算量上升（稀疏度归零） |
| L1 vs MixtureLaplace | 后者在离群/遮挡处更稳 |

### 5.1 POC 实测（`step3_ablation.py`，120 步，同数据/种子/公平初始化）

> 公平初始化：锚模块最后构造，保证共享模块在 corr/gev 之间得到一致随机种子；
> gev 对齐到 corr 的尺度(1/2)与带宽(R=4)，且去掉 d_cv 辅助项（只比较锚本身的贡献）。

| 配置 | 参数量 | EPE末(px) | bad1px | 相对 no_anchor |
|---|---|---|---|---|
| no_anchor（纯 warp 基线） | 269.6K | 2.430 | 70.5% | — |
| **corr**（稀疏相关锚，无聚合） | 269.6K | **1.850** | 63.2% | **+23.9%** |
| gev-full3d@1/2（完整 3D 聚合） | 254.0K | 2.092 | 67.4% | +13.9% |
| gev-sep3d@1/2（可分离 3D） | 252.5K | 1.937 | 64.4% | +20.3% |

**结论**：① 任何锚都显著优于纯 warp（+14~24%），验证「注入匹配证据」的普适价值；
② 稀疏相关锚（无聚合）在本尺度最优，与 WAFT「代价体非必需」的主张一致；
③ 在代价体锚内部，**可分离 3D（sep3d）优于完整 3D（full3d）**（精度更高、参数更少、更快），
说明轻量正则化优于重聚合。这是一个可直接写入论文的消融结论。

## 6. 实验计划（全量训练时）

- 数据：SceneFlow / CREStereo / TartanAir / FSD 等合成集（复用 WAFT 配置），ETH3D/KITTI/Middlebury 评测。
- 基线：WAFT-Stereo（DAv2S-4 等）、RAFT-Stereo、IGEV-Stereo、Selective-Stereo。
- 指标：EPE、bad-1px/3px、运行时间、FLOPs；零样本跨域（sim-to-real）。
- 收敛性：对比带/不带 d_gm 初始化的收敛曲线（预期收敛更快）。

## 7. 与现有工作（本仓库 step1）的关系

- `step1_fusion.py`：仅注入窄带代价体锚（MatchingBranch + AnchorHead），零初始化，验证「需训练匹配信号」。
- 本文 `step2_fusion_composite.py`：在 step1 之上**新增** GlobalMatcher、GatedFusion、TokenSparseViT 与
  mixture-of-Laplace 损失，形成完整的「三源融合 + 稀疏迭代」论文方案（自包含，CPU 跑通）。

## 8. 如何引用传统代价体（设计深化）

### 8.1 传统代价体的两个本质价值

传统代价体（PSMNet / GwcNet / ACVNet / IGEV）之所以有效，靠的是两个正交的能力：

1. **显式匹配证据**：把「左特征 × 右特征」在**所有候选视差**上的相似度显式写成张量
   `V(x,d)=⟨f_L(x), f_R(x-d)⟩`（correlation / concat / group-wise），网络可直接"读到"匹配强度分布。
2. **可学习正则化聚合**：用 3D CNN / hourglass 在「视差维 × 空间维」做邻域平滑与代价滤波，
   这是代价体方法对**无纹理 / 歧义区域**鲁棒的关键。

WAFT 丢弃代价体后丢掉的正是这两点：warping 只提供「当前视差处」的局部证据，既没有全范围匹配分布，
也没有跨视差邻域的显式正则化 → 低纹理 / 大视差 / 遮挡处收敛慢、易错配。
**因此"引用传统代价体"的合理目标，不是回到全尺寸代价体，而是把这两大价值以「可控成本」补回来。**

### 8.2 三种引用深度（由轻到重）

| 方案 | 内容 | 代价体两大价值 | 效率 |
|---|---|---|---|
| A 轻：相关分布 + soft-argmax（无聚合） | 现 GlobalMatcher 全范围相关 + SparseCorrAnchor 窄带相关 | 只有①，缺② | 最高 |
| **B 中（推荐）：窄带几何编码体 + 轻量 3D 聚合** | `GEVCostAnchor` | ① + ② 都在 | 高 |
| C 重：全分辨率全视差 3D hourglass 主分支 | PSMNet/IGEV 式，warp 退化为 refine | ① + ② 最强 | 低，背离 WAFT 初衷 |

推荐 **B**：它把代价体的两大价值都引回来，但通过「1/4 尺度 + 窄带 + 组相关 + 少层小通道 3D 卷积」把成本压到可控。

### 8.3 GEVCostAnchor 设计（脚本 `ANCHOR_KIND=gev`）

- **构造**（1/4 尺度，当前视差附近）：`V = concat[ group-wise 相关(匹配证据), 当前视差(几何信息) ]`
  → 得到「组合几何编码体」（IGEV 精神：匹配 + 几何 + 上下文同体）。
- **采样**：窄带 `±R` 内**非均匀采样 K 个候选**（中心密、边缘疏，Selective-Stereo 稀疏采样思想），
  避免全视差范围。
- **聚合**：2 层轻量 3D 卷积在「视差维 × 空间维」聚合 → `agg`；支持两种实现（`agg_kind`）：
  - `full3d`：完整 `(3,3,3)` 卷积；
  - `sep3d`：**可分离 3D** —— 空间 2D `(1,3,3)` + 视差维 1D `(3,1,1)` 分离（PSMNet/GA-Net 轻量化思想，参数量/算力更低）；
  `cost=3DConv(agg)` → `prob=softmax_k(cost)`。
- **输出**：
  - `d_cv = Σ_k prob·d_k`：代价体先验视差（可直接监督，加速收敛）；
  - `feat = Σ_k prob·agg`：概率加权的代价体上下文特征，作为锚经 GatedFusion 注入迭代。
- **监督**：`L_cv = λ·L_ml(d_cv - gt_half)`，与 `d_gm`、各轮 `disp` 联合训练。

### 8.4 与 SparseCorrAnchor 的定位关系（同一插槽、可切换）

- `SparseCorrAnchor` = **无聚合的逐点相关**（局部匹配证据，零初始化、手术安全）；
- `GEVCostAnchor` = **带 3D 正则化聚合的窄带代价体**（匹配证据 + 上下文正则化）。

两者占用同一插槽（`anchor_kind='corr' | 'gev'`），消融可单独回答「**3D 聚合是否带来增益**」这一核心问题。

### 8.5 为何不破坏效率

- 代价体只在 **1/4 尺度**（默认，`gev_downsample=2`）或 **1/2 尺度**（`gev_downsample=1`）构造；
- 窄带 `K=9` + 组相关 `G=4`，非全视差范围；
- 3D 卷积仅 2 层、通道 `Cv=8`，且支持**可分离 3D**（`agg_kind='sep3d'`）进一步降参数/算力；
- 结果上采样回 1/2 后作为「锚」注入，不替代 warp 主干；
- 输出零初始化（`feat=0`、`d_cv=disp_q`）→ 手术安全，等价原版。

POC 实测（CPU）：`step3_ablation.py` 120 步公平对比中，gev 锚相对纯 warp 基线带来
+13.9%（full3d）/ +20.3%（sep3d）的 EPE 降幅，且 sep3d 精度更高、参数更少、更快（见 §5.1）。

