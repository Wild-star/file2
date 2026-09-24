# 对照实验方案：FusionWarp 作为 WAVE-Stereo 的差异化变体

> 诚实定位：本设计的核心 insight（相关体与 warp 互补）已被 **WAVE-Stereo**
> （arXiv:2607.13674，2026-07）抢先，故本文**不再主张「统一相关与 warp」这一原创贡献**，
> 转而回答一个更聚焦、也更有意义的问题：

> **在「相关 + warp 互补融合」这个共同框架下，本设计的三个实现选择（GlobalMatcher
> 全局匹配初始化、TokenSparseViT 稀疏解码、GatedFusion 门控融合）相对 WAVE-Stereo
> 的对应选择（2D 代价聚合初始化、ConvGRU、concat 融合），是否带来可测量的增益？**

## 1. 对比的公平性前提

- **共同框架**：相关检索 + warp 残差 + 迭代更新（双方已一致，这是 WAVE-Stereo 的贡献，我们引用而非重复）。
- **差异仅在被消融的轴**上；其余（encoder 尺度、损失、上采样、数据）尽量对齐。
- **POC 层**：自包含合成数据、CPU、同种子、同步数，用于方向信号；
  **最终结论**必须在 WAVE-Stereo 的 9 数据集全量训练设定（8×A100）下复验。

## 2. 消融维度与配置

| 配置 | 初始化 | 迭代单元 | 融合 | 相关锚 | 用途 |
|---|---|---|---|---|---|
| `wave` | Corr2DInit（相关 2D 聚合 + soft-argmin） | ConvGRU | concat | 窄带相关（GWCE corr 分支） | **baseline** |
| `fusion` | GlobalMatcher（全局交叉注意力 + 全范围相关） | TokenSparseViT | GatedFusion | SparseCorrAnchor | **本设计** |
| `fusion_noanchor` | GlobalMatcher | TokenSparseViT | GatedFusion | 无（zeros） | 测「锚」贡献 |
| `fusion_nosparse` | GlobalMatcher | TokenSparseViT(sparse=False) | GatedFusion | SparseCorrAnchor | 测「稀疏」贡献 |

> 单因子对照规则：每个 `fusion_*` 只改动一个轴，其余保持 `fusion` 默认，从而把
> `fusion` 相对 `wave` 的整体差距**归因到具体轴**。

## 3. 度量与判定

- 主指标：末步 EPE（px）、bad-1px（%）；副指标：参数量、每步耗时。
- 判定（POC 方向信号，非最终结论）：
  - `fusion` EPE < `wave` EPE → 本设计的实现组合在 POC 层面**不劣于** WAVE 风格；
  - `fusion_noanchor` / `fusion_nosparse` 与 `fusion` 的差值 → 锚 / 稀疏各自的边际贡献；
  - 若 `fusion` 无优势或更差 → 诚实结论：本设计的差异点是**无效/有害变体**，论文定位降级为
    「对 WAVE-Stereo 的复现 + 无效变体排除」，同样是有价值的负结果。

## 4. 全量验证计划（POC 之后，真实设定）

1. 复现 WAVE-Stereo（官方代码，8×A100，9 数据集）作为唯一 baseline；
2. 在 WAVE 代码上实现三个替换开关：`init=gm|corr2d`、`decoder=sparsevit|convgru`、`fusion=gate|concat`；
3. 逐因子消融 + 5 基准零样本（Middlebury/ETH3D/KITTI12/KITTI15/Booster）；
4. 报告每个轴的 ΔEPE / ΔD1-all / Δlatency，并诚实声明哪些轴有效、哪些无效。

## 5. 结果记录

### 5.1 POC 层结果（`step4_comparison.py`，80 步，合成数据，CPU）

| 配置 | 参数量 | EPE末(px) | bad1px | 相对 wave |
|---|---|---|---|---|
| `wave`（baseline） | 234.0K | 4.045 | 84.6% | — |
| `fusion`（本设计） | 269.6K | **2.050** | 64.8% | **+49.3%** |
| `fusion_noanchor` | 269.6K | 2.287 | 70.1% | +43.5% |
| `fusion_nosparse` | 269.6K | 2.688 | 73.9% | +33.6% |

单因子边际贡献（fusion 相对 wave 的 +49.3% 的分解）：

- **token 稀疏**（`fusion` 2.050 → `fusion_nosparse` 2.688）：贡献最大（约 −0.64 EPE）；
- **相关锚**（`fusion` 2.050 → `fusion_noanchor` 2.287）：次之（约 −0.24 EPE）；
- **GlobalMatcher 初始化**：残余贡献（两个消融都去掉后仍优于 wave，说明初始化 + 门控 + 迭代骨架的差异也起作用）。

### 5.2 诚实边界（务必读）

1. `wave` 是**自包含极简镜像**（ConvGRU + Corr2DInit + concat），**不是 WAVE-Stereo 官方实现**
   （官方另含 MobileNetV2/LightStereo 2D 聚合、12 轮迭代、PGCP）。因此上表**不能**解读为
   「本设计优于 WAVE-Stereo」，只能解读为「本设计的三个差异点在 POC 层各自有正向边际贡献」。
2. 这是合成数据、80 步、单种子的**方向信号**，不是最终结论。
3. 最终结论必须在 WAVE-Stereo 官方代码 + 9 数据集全量设定下复验（§4）。

### 5.3 结论（POC 层）

三个差异点（GlobalMatcher 初始化 / token 稀疏 / 门控融合 + 窄带锚）在 POC 层均为**有益的变体**，
其中 token 稀疏的边际增益最显著。这为「FusionWarp 作为 WAVE-Stereo 的差异化变体」提供了
初步的正向证据，但远不足以构成独立方法；论文主张仍需以全量复现为准。

- POC 原始数据：`step4_comparison.json`。
- 全量层结果：待 WAVE-Stereo 复现完成后回填。

## 6. 引用

- WAVE-Stereo: Warp-Aligned Volume Encoding for Stereo Matching, arXiv:2607.13674 (2026).
- WAFT-Stereo（本设计基线）: arXiv:2603.24836.
