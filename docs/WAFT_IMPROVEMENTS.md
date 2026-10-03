# 相对原始 WAFT 的改进清单与回归决策

> 目的：诚实梳理本仓库在原始 WAFT-Stereo（warp-only，无代价体）之上加过的所有模块，
> 明确哪些已证明无效，据此**回归原始 WAFT**，再以「代价体最小注入」重新起步。

## 1. 原始 WAFT 结构（baseline）

```
encoder(stack[img1,img2]) → fmap1, fmap2, net
初始化(prop): prop_proj(cat[fmap1,fmap2]) → VitIter → prop_bins_head
             → softmax → disp = Σ_bins bin·idx            # 分类式初始化，无代价体
迭代(delta): disp.detach()
             warped = disp_warp(fmap2, disp)               # 唯一几何线索
             net = delta_proj(cat[fmap1, warped, net, disp])
             net = delta_decoder(VitIter)
             disp += delta_disp_head(net)
             disp_up = convex_upsample(disp*2, mask)
```

核心特征：**纯 warp（feature warping），无代价体 / 无相关体 / 无显式匹配检索**。

## 2. 改进清单（相对 WAFT 加过的模块）

| # | 模块 | 相对 WAFT 的改动 | 开关/位置 |
|---|---|---|---|
| 1 | `GlobalMatcher` | 用全局注意力 + 全范围相关替代 prop bins 分类初始化 | `use_global_init` |
| 2 | `DPI` 深度 warm-start | 单目深度 → 视差增量（零初始化） | `use_dpi` |
| 3 | `MatchingBranch`/`MatchFeat` | 新增独立匹配特征分支 | 随锚启用 |
| 4 | `SparseCorrAnchor` | 窄带 group-wise 相关锚，注入迭代 | `anchor_kind='corr'` |
| 5 | `GEVCostAnchor` | 窄带几何代价体锚（3D/可分离 3D 聚合） | `anchor_kind='gev'` |
| 6 | `GatedFusion` | 门控融合「局部锚 + 全局上下文 g_feat」 | `use_gated_fusion` |
| 7 | `TokenSparseViT` | token 级稀疏解码器（替代 VitIter） | 仅 POC（step2） |
| 8 | `MixtureLaplaceLoss` | 混合拉普拉斯损失 | 仅 POC（step2） |
| 9 | OT 全局匹配初始化 | Sinkhorn OT 替代 softmax | step6（光流 FlowIt） |
| 10 | 逐像素 σ 引导损失 | aleatoric 不确定性引导 | step5（光流 U²Flow） |

## 3. 结论：前面方法的无效性已由充分训练证明（精度大幅下降）

> ⚠️ **关键更正**：今日对话前的充分训练工作已证明——本清单所列改进（GlobalMatcher/锚/门控/
> token 稀疏/OT/σ 等）在真实充分训练下**无效、精度大幅下降**。这是既定结论，本仓库以它为准。

80 步合成数据 POC（`step8`~`step10`）曾给出「fusion 优于 WAFT −54%」的**假象**：

| 模型 | EPE mean±std | 说明 |
|---|---|---|
| 原始 WAFT（纯 warp-only） | 4.758 ± 0.460 | baseline |
| fusion（全模块） | 2.203 ± 0.123 | 80 步 POC 显示「−54%」，**短训练假象，充分训练已推翻** |

**为什么 POC 给出相反结论**：80 步未收敛 + 合成分数据下，更复杂的模型恰好处在更低的中间
loss 点，不能代表充分训练后的最终精度；深度模块的真实效果**须充分训练才能判定**。

**教训（写入训练协议）**：以短训练 POC 判定深度模块有效/无效是无效证据，已废止；一切有效性
判断以充分训练为准。

## 4. 回归决策

1. **代码回归原始 WAFT**：`algorithms/waft.py` 的融合模块默认关闭即原版；POC 侧新增
   `step10_waft_baseline.py`（自包含纯 warp-only WAFT，CPU 可跑，EPE 4.758±0.460）作干净起点。
2. **后续聚焦「代价体 + WAFT 最小注入」**：不再「一次加 7 个模块」，而是**在原始 WAFT 上
   只加一个代价体注入点**（其余 prop bins 初始化 / VitIter / warp 全部保持原样），
   单因子、多种子、可归因。

## 5. 代价体 + WAFT 最小注入的候选方案（待调研后定）

- 候选 A：代价体特征 **concat 进 delta_proj 输入**（`cat[fmap1, warped, net, disp, cost_feat]`）；
- 候选 B：代价体 **作为 prop 初始化的辅助监督**（训练用、推理可去）；
- 候选 C：代价体 **替换 warped 残差**（把 warp 对齐特征换成代价体匹配特征，或两者互补）；
- 候选 D：**多尺度代价体**（1/4 代价体 + 1/2 warp，跨尺度互补）。

下一步：调研 CVPR/ICCV/ECCV/NeurIPS 2024–2026 的双目立体匹配与光流估计，验证/修正候选方案。

## 6. 候选 A 实测结果（`step11_cost_injection.py`，80 步 × 3 种子）

| 配置 | EPE mean±std | 逐种子 |
|---|---|---|
| WAFT（warp-only） | **4.758 ± 0.460** | 4.125 / 4.943 / 5.205 |
| WAFT+Cost（窄带相关 concat 注入） | 6.061 ± 2.067 | 3.732 / 5.696 / 8.755 |

**负结果**：窄带相关 `cost_feat` 直接 concat 进 delta_proj **不稳健、甚至有害**（mean 差 −27%，
std 从 0.46 暴涨到 2.07，即引入严重初始化敏感）。

**根因（关键诊断）**：`warped_fmap2 = disp_warp(fmap2, disp)` 本身已是「当前视差下的单点对齐
（隐式代价）」，而窄带相关给的也是「当前视差 ±R 的匹配证据」——**两者信息高度冗余、不互补**，
concat 进去只是给 delta_proj 加了需要学习忽略的噪声。这解释了为何 WAVE-Stereo 有效：它的
correlation 提供**多候选匹配证据**（与单点 warp 互补），且用专门的分支编码（GWCE）+ ConvGRU，
而非简单 concat。

**教训**：代价体注入要有效，必须满足「**与 warp 互补**」（信息不冗余）且「**有专门融合机制**」
（非 concat）。下一步方向：
- 候选 B：cost_feat 改为 **epipolar 全局相关**（全视差范围，提供 warp 单点没有的全局匹配证据）；
- 候选 C：cost_feat 提供**不同模态**（频域高/低频 或 空间×视差解耦），与 warp 的光度信息互补；
- 融合：从 concat 升级为门控/注意力融合。

> 注：以上「候选 A 根因诊断」是设计层面的分析（冗余/互补/融合机制），**不构成有效性判据**；
> 任何候选的最终有效性须在充分训练下验证。
