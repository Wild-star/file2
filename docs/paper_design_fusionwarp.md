# FusionWarp-Stereo：多源证据与置信引导的迭代双目匹配

## —— 模块迁移缝合设计文档（v2，聚焦"如何缝合别人的优秀模块"）

> 本设计的主旨不是"我发明了哪些新模块"，而是回答：**在 WAFT 式纯 warp 迭代框架上，
> 如何把成熟文献中经过充分训练验证的优秀机制，按一条主线迁移、缝合，形成一个有机整体**。
> 每个模块都标注来源、迁移的机制、缝合位置、解决的短板。深度模块的有效性须经充分训练
> 验证，本设计不依赖任何短训练 POC 指标下结论。

---

## 0. 设计定位（一句话）

WAFT-Stereo 证明了「纯 feature-warping、无代价体」的可行性，但其信息流只有**单点 warp 对齐**
一路；本设计把它升级为「**多源匹配证据 + 置信引导的迭代匹配**」：每路证据、每个引导机制
都从成熟文献迁移而来，缝合在 WAFT 的迭代主干的明确位置上。

## 1. 设计哲学：一条主线、三类短板、八个迁移模块

主线：**证据 → 置信 → 效率**（Evidence → Confidence → Efficiency）。

- **证据流（Evidence）**：WAFT 只有单点 warp，缺「多候选匹配证据」与「全局定位」；
- **置信流（Confidence）**：有了多源证据，需学习「信谁、在哪里信」；
- **效率流（Efficiency）**：证据与置信带来额外算力，需「选择性投入」保持 WAFT 的高效。

三类短板、八个模块全部从已发表文献迁移（无一是本设计原创），缝合逻辑见下表。

## 2. 模块迁移缝合总表（核心）

| # | 模块 | 迁移来源（谁·机制） | 缝合位置 | 解决的短板 | 主线 |
|---|---|---|---|---|---|
| A1 | GlobalMatcher | GMFlow (2021) / SEA-RAFT (2024)：全局匹配 + 直接初始回归 | 替代 WAFT 的 `prop_bins` 分类初始化 | 粗定位（大视差/低纹理） | 证据 |
| A2 | CostAnchor | RAFT-Stereo (2021) / IGEV (2023)：窄带相关/几何代价体 | 迭代内、与 `warp(f2,disp)` 并列 | 多候选匹配证据（单点 warp 缺失） | 证据 |
| A3 | 解耦代价体聚合 | DBStereo (2025)：4D 代价体空间×视差 2D 解耦 | A2 的聚合层 | 代价体效率（无 3D 卷积） | 证据 |
| B1 | GatedFusion | ACVNet (2022) / CREStereo (2022)：空间门控融合 | A2 锚与全局上下文之间 | 局部 vs 全局证据权衡 | 置信 |
| B2 | Uncertainty | U²Flow (2026) / URS-Stereo (2026)：逐像素 aleatoric 不确定性 | 迭代隐状态上的 σ 头，复用三处 | 难例自适应（遮挡/无纹理） | 置信 |
| B3 | GlobalContext | GREAT-Stereo (ICCV 2025)：SA(空间)+MA(epipolar)+VA(体积) 注意力 | 迭代内、全局上下文注入 | 无纹理/重复纹理歧义 | 置信 |
| C1 | TokenSparseViT | DynamicViT (NeurIPS 2021)：token 级动态稀疏化 | 替代 WAFT 的 `VitIter` 全量 token | 迭代算力冗余 | 效率 |
| C2 | 代价体训练蒸馏 | Removing Cost Volumes (ICCV 2025)：代价体训练后移除 | A2 的训练/推理分离 | 推理代价体冗余 | 效率 |

## 3. 逐模块迁移设计

> 统一模板：来源 → 解决的短板 → 缝合位置 → 设计细节 → 协同关系。

### 3.1 证据流（Evidence）

#### A1 GlobalMatcher —— 全局匹配初始化
- **来源**：GMFlow 的「transformer 全局匹配」+ SEA-RAFT 的「direct 初始回归」。
- **短板**：WAFT 的 `prop_bins` 在局部感受野内做 bins 软分类，缺乏全局定位，大视差/低纹理下易错。
- **缝合**：替换 `prop_proj/prop_decoder/prop_bins_head` 为「1/8 池化 → 左右交叉注意力 → 全范围相关
  `C(x,d)=⟨l(x), r(x-d)⟩` → soft-argmax 得 `d_gm`」；`d_gm` 作为初始视差，交叉注意力特征投影为全局上下文 `g_feat`。
- **协同**：`g_feat` 同时喂给 B1（门控）与 B3（全局上下文），是「置信流」的输入之一。

#### A2 CostAnchor —— 窄带代价体锚
- **来源**：RAFT-Stereo 的「迭代相关体检索」+ IGEV 的「几何编码体」。
- **短板**：`warped_fmap2 = warp(f2, disp)` 只是**当前视差下的单点对齐**，无多候选匹配证据（候选 A 实测
  亦证明：窄带相关若与 warp 简单 concat 则冗余，须作为**独立锚 + 门控**注入，见 B1）。
- **缝合**：在独立匹配特征 `m1,m2` 上，于当前 `disp` 的 `±R` 窄带做 group-wise 相关，经**零初始化**投影
  得锚特征 `anchor`（零初始化保证加载预训练 WAFT 后行为等价原版）。
- **协同**：`anchor` 不直接 concat 进 `delta_proj`，而是先经 B1 门控与全局上下文融合——这是与
  「候选 A 失败」的关键区别。

#### A3 解耦代价体聚合 —— 空间×视差 2D 解耦
- **来源**：DBStereo 的「4D 代价体解耦为空间维 + 视差维，纯 2D 卷积聚合」。
- **短板**：代价体 3D 卷积是效率瓶颈，与 WAFT 轻量定位矛盾。
- **缝合**：A2 的窄带代价体聚合用「空间 2D 卷积（spatial）+ 视差方向 1D 卷积（disparity）」解耦，
  替换 3D 正则化，保持轻量。
- **协同**：与 C2（训练蒸馏）一起构成「代价体只在训练时充分、推理时廉价」的完整效率设计。

### 3.2 置信流（Confidence）

#### B1 GatedFusion —— 空间门控融合
- **来源**：ACVNet 的 attention gate / CREStereo 的融合门控。
- **短板**：局部锚（纹理处可靠）与全局上下文（无纹理处可靠）的可靠域互补，需**空间自适应**选择。
- **缝合**：`g = σ(Conv([anchor, g_feat]))`，`fused = g·anchor + (1-g)·g_feat`，`fused` 作为额外一路
  注入 `delta_proj` 输入。
- **协同**：是「候选 A 简单 concat 失败」的修正——从「无选择拼接」升级为「可学习门控选择」。

#### B2 Uncertainty —— 逐像素不确定性引导
- **来源**：U²Flow 的「联合估计 flow + aleatoric 不确定性」+ URS-Stereo 的「不确定性引导残差搜索」。
- **短板**：遮挡/无纹理是难例，固定损失/固定搜索半径无法自适应。
- **缝合**：迭代隐状态上加 `σ = softplus(conv(net))`，复用三处：
  ① 调制 B1 门控（低置信处更偏向全局上下文）；② 自适应 A2 窄带搜索半径 `R`；③ 损失权重（难例降权）。
- **协同**：是「置信流」的**统一来源**——σ 同时服务 B1、A2、损失，而非独立堆砌。

#### B3 GlobalContext —— 全局上下文注意力
- **来源**：GREAT-Stereo 的 SA（空间）+ MA（epipolar 匹配）+ VA（体积）三注意力，即插即用注入迭代。
- **短板**：WAFT 迭代无全局上下文传播，无纹理/重复纹理区域无法从远处可靠区「借用」信息。
- **缝合**：在迭代内对隐状态做 epipolar 方向（沿极线）+ 空间方向的轻量注意力，得到全局上下文，
  与 A1 的 `g_feat` 合并后供 B1 门控。
- **协同**：与 B1/B2 构成完整的「全局上下文 → 门控选择 → 难例自适应」置信链。

### 3.3 效率流（Efficiency）

#### C1 TokenSparseViT —— token 级选择性更新
- **来源**：DynamicViT（NeurIPS 2021, 2106.02034）的动态 token 稀疏化（Selective-Stereo 的核心
  实为多频率融合 SRU+CSA，已归入 B1 佐证链，不属 token 稀疏）。
- **短板**：WAFT 的 `VitIter` 对全量 token 做注意力，但逐轮视差增量高度稀疏（本仓库 P0 诊断）。
- **缝合**：patch 化后 router 预测 saliency，`gate = σ(sal)`，`h' = tok + gate⊙(Attn(tok)-tok)`；
  训练加 `λ·mean(gate)` 鼓励稀疏，硬 top-k 版本推理跳过低 saliency token 的注意力。
- **协同**：稀疏性可由 B2 的 σ 调制（低置信 token 才充分更新），与「置信流」耦合。

#### C2 代价体训练蒸馏 —— 训练充分、推理移除
- **来源**：Removing Cost Volumes（ICCV 2025）的「代价体训练后重要性下降、可蒸馏移除」。
- **短板**：A2 代价体在推理时仍有开销，而实验表明代价体在训练充分后趋于冗余。
- **缝合**：A2 在**训练**时作为完整锚 + 辅助监督（`d_cv`），训练后期用蒸馏/剪枝使网络学会
  「不依赖代价体也能预测」，**推理**时移除 A2 分支。
- **协同**：与 A3（轻量聚合）共同保证「代价体不破坏 WAFT 高效」。

## 4. 模块协同逻辑（为什么不是堆砌）

八个模块沿三条主线形成**三条闭环**，而非并列：

1. **证据闭环**：A1 全局初始 → A2/A3 局部代价体 → 两者经 B1 门控融合成「多尺度证据」；
2. **置信闭环**：B3 全局上下文 + B2 逐像素 σ → 共同调制 B1 门控与 A2 搜索半径；
3. **效率闭环**：C1 稀疏（按 B2 置信度选择 token）+ C2 蒸馏（按训练进度移除代价体）。

**一个统一的迭代更新**（缝合后的完整前向）：

```
d_gm, g_feat = A1_GlobalMatcher(f1, f2)        # 全局初始 + 全局上下文
disp = d_gm ; net = 0
for itr in 1..T:
    disp     = detach(disp)
    anchor   = A2_CostAnchor(m1, m2, disp)      # 窄带代价体，A3 解耦聚合
    g_ctx    = B3_GlobalContext(net, disp)      # epipolar/空间全局注意力
    fused    = B1_GatedFusion(anchor, g_feat+g_ctx)   # 门控选择局部 vs 全局
    σ        = B2_sigma_head(net)               # 逐像素不确定性
    warped   = warp(f2, disp)
    x        = cat[f1, warped, fused, net, disp]
    net      = delta_proj(x)
    net, gate= C1_TokenSparseViT(net, conf=σ)   # 置信调制的稀疏更新
    Δdisp    = disp_head(net)
    disp     = disp + Δdisp
    disp_up  = convex_upsample(disp*2, mask)
# 推理时：C2 移除 A2 分支，仅保留 warp + 全局上下文
```

## 5. 训练协议（充分训练，杜绝短 POC 判定）

- **数据**：SceneFlow（FlyingThings3D 主训）+ 多域（KITTI/Middlebury/ETH3D/Booster）零样本评测；
- **充分收敛**：完整 epoch、学习率 warmup + cosine 衰减、与基线同协议对比；
- **统计严谨**：≥3 种子报告 mean±std + 配对 t 检验（本仓库已建立 `step8/step9` 多种子协议）；
- **消融**：每个模块**单因子**消融（其余保持基线），而非一次性堆叠；
- **明确不做**：以 80 步合成数据 EPE 判定模块有效/无效（这是无效证据，已废止）。

## 6. 创新性说明（诚实：迁移缝合 + 差异化主张）

- **本设计不主张任何单个模块的原创**：A1~C2 均来自已发表文献（见 §2 来源列）。
- **主张的是「缝合」本身**：在 WAFT 式纯 warp 迭代主干上，把「证据-置信-效率」三线闭环缝合成
  一个可训练、可消融的整体，并回答每个模块**在哪、为何、如何**接入。
- **与 WAVE-Stereo（prior work，arXiv:2607.13674）的差异化**：WAVE 用 ConvGRU + GWCE 三分支 +
  PGCP；本设计用 **ViT 迭代（C1）+ 门控选择（B1）+ 不确定性引导（B2）+ 代价体蒸馏（C2）** 四点为差异，
  需在全量训练下逐点对照验证（对照方案见 `docs/COMPARISON_PLAN.md`）。
- **定位**：WAVE-Stereo 的**差异化变体 / 消融研究**，而非独立新范式。
