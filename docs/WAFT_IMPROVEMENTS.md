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

## 3. 多种子实验结论（80 步 × 3 种子，`step8`/`step9`/`step10`）

**整体 fusion 相对原始 WAFT 有显著增益（多种子）：**

| 模型 | EPE mean±std | 相对 WAFT |
|---|---|---|
| 原始 WAFT（纯 warp-only，`step10`） | 4.758 ± 0.460 | baseline |
| fusion（全模块，`step9`） | 2.203 ± 0.123 | **−53.7%（显著）** |
| fusion_nosparse（去 token 稀疏） | 2.350 ± 0.271 | −50.6% |
| fusion_noanchor（去相关锚） | 2.962 ± 0.966 | −37.7% |

**澄清（重要）**：此前「所有改进无效」的说法**不准确**。准确的是：
- **整体 fusion 相对 WAFT 显著有效（−54%，std 不重叠）**，且去掉单个模块后仍显著优于 WAFT；
- 但**无法归因到单个模块**——单因子消融（token 稀疏、锚）存在种子反转，边际贡献不稳健；
- #9（OT）/#10（σ）两个光流模块**确实无稳健正增益**。

根因是「一次堆叠过多模块、改动过大无法归因」。正确后续：**回归 WAFT 后，做单因子代价体
注入**，逐项找出真正贡献的模块。

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
