# FusionWarp-Stereo：Warp-代价体互补的补充式融合（学习 WAVE GWCE）

## —— 论文设计文档（v3，补充式 GWCE 旁路）

> 本设计回答两个问题：**(1) 代价体融合是替换还是补充 WAFT？** 答案是**补充**——WAFT 原始
> 结构一字不动，代价体作为旁路以门控残差注入。**(2) 怎么融合 warp 与代价体？** 学习
> WAVE-Stereo 的 GWCE（三路并行编码 + 融合）与 PCW-Net 的 warping volume（用视差缩小搜索范围）。
>
> 深度模块的有效性须经充分训练验证，本设计不依赖任何短训练 POC 指标下结论；前面 fusion-full
> 的「替换式 + 多模块堆叠」已被充分训练证明无效（精度大幅下降），本设计是对其失败教训的纠正。

---

## 0. 设计定位（一句话）

WAFT-Stereo 的纯 warp 迭代只有「单点对齐」一路匹配线索，缺「多候选匹配证据」；本设计在
WAFT 迭代主干**之外**加一个 **GWCE 式三路并行编码旁路**，把 correlation（代价体）+ warp（对齐）
+ disparity prior（视差先验）三路互补线索融合后，以**门控残差**补充进迭代——**补充而非替换**。

## 1. 核心问题与融合范式

### 1.1 为什么「替换式 + 堆模块」失败（前面教训）

fusion-full 的失败根因：**替换** WAFT 三处核心（GlobalMatcher 换 prop bins、TokenSparseViT 换
VitIter、MixtureLaplace 换损失）+ **一次堆 5 个模块**，破坏了 WAFT 的 warp 自回归一致性。
结论：融合必须**补充**（保留原结构），且**收敛为一个统一旁路**（非零散开关）。

### 1.2 学到的两个融合范式

**① WAVE-Stereo（arXiv:2607.13674）GWCE —— 三路并行编码 + 融合**：

```
x_c = Enc_c(c)                        # correlation 分支：1×1 + 3×3，保留候选歧义
x_d = Enc_d(d)                        # disparity prior 分支：7×7 大核 + 3×3
f̃_R = Warp(f_R, d);  x_w = Enc_w([f_L, f̃_R])   # warp 对齐分支：两个 3×3
m_t = [Fusion([x_c, x_d, x_w]), d]             # 三路融合 → 送 ConvGRU
```

关键：**每路独立编码，再 Fusion**——不是直接 concat 原始信号（这正是候选 A 失败的根因：
直接 concat 窄带相关，缺「独立编码 + 融合层」）。

**② PCW-Net（ECCV 2022 Oral）warping volume —— 视差缩小搜索范围**：
用当前视差把代价体搜索范围从全范围缩小到窄带，粗→细。

### 1.3 补充式结构决策

| 维度 | 替换式（已证失败） | 补充式（本设计） |
|---|---|---|
| WAFT `delta_proj`/`delta_decoder` | 换成 UECF | **一字不动** |
| prop bins 初始化 / VitIter / 损失 | 替换 | **一字不动** |
| 代价体融合 | 塞进 `delta_proj` 输入 | **旁路门控残差** `net = net + gate(m)` |
| 可加载预训练 WAFT | 否（结构变了） | **是**（`gate` 零初始化 → 等价原版） |

## 2. 方法：补充式 GWCE 三路并行编码旁路

### 2.1 完整前向

```
encoder(stack[img1,img2]) → fmap1, fmap2, net
# WAFT 原始 prop 初始化（分类式，不动）
prop_bins → softmax → disp = Σ_bins bin·idx

for itr in 1..T:
    disp   = detach(disp)
    warped = disp_warp(fmap2, disp)              # WAFT 原 warp

    # === 新增旁路：GWCE 式三路并行编码（学习 WAVE + PCW）===
    x_c = Enc_c(corr_lookup(m1, m2, disp))       # ① correlation：窄带检索（PCW warping volume）
    x_w = Enc_w(cat[fmap1, warped])              # ② warp 对齐（复用 warped，WAFT 原信号）
    x_d = Enc_d(disp)                            # ③ disparity prior（7×7 大核）
    m   = Fusion(cat[x_c, x_w, x_d])             # 三路融合（1×1 卷积）

    # === WAFT 原始 delta 迭代（一字不改）===
    net = delta_proj(cat[fmap1, warped, net, disp])
    net = delta_decoder(net)                     # VitIter 原样

    # === 补充：门控残差注入 ===
    net = net + gate(m)                          # 零初始化 → 加载预训练 WAFT 后等价原版

    Δdisp = disp_head(net);  disp = disp + Δdisp
    disp_up = convex_upsample(disp*2, mask)
```

### 2.2 三路分支设计

| 分支 | 输入 | 编码器 | 提供的信息 | 迁移来源 |
|---|---|---|---|---|
| ① correlation | `corr_lookup(m1,m2,disp)` 窄带相关体 | 1×1 + 3×3 卷积 | 多候选匹配证据 + 歧义分布 | RAFT-Stereo / IGEV / PCW |
| ② warp 对齐 | `cat[fmap1, warped]` | 两个 3×3 卷积 | 跨视角对齐残差 | WAFT / WAVE |
| ③ disparity prior | `disp` | 7×7 大核 + 3×3 | 当前几何状态的空间分布 | WAVE |

### 2.3 融合与注入

- **Fusion**：三路 concat 后 1×1 卷积投影到隐维度（学 WAVE GWCE 的 `Fusion`）。
- **门控注入**：`gate(m)` 输出残差，`gate` 用**零初始化**的 1×1 卷积实现 → 加载预训练 WAFT 后
  `gate(m)≡0`，行为**严格等价原版**，消融干净、可单因子归因。

## 3. 迁移依据（每一路的论文支撑）

- **① correlation 窄带检索**：RAFT-Stereo（迭代相关体检索）+ PCW-Net（warping volume 用视差
  缩小搜索范围）+ IGEV（几何编码体）。→ 提供 WAFT 单点 warp 缺失的「多候选匹配证据」。
- **② warp 对齐**：WAFT 原信号 + WAVE 的 cross-view warping alignment branch。→ 与 correlation
  **互补**（WAVE 核心洞察：相关保留候选歧义，warp 把问题转为局部残差修正）。
- **③ disparity prior**：WAVE 的 disparity prior branch（7×7 大核感知视差空间分布）。
- **三路融合**：WAVE GWCE 的 `Fusion`（并行编码后统一融合，而非简单 concat）。
- **门控残差注入**：ACVNet / Selective-Stereo（CSA）的「注意力门控融合异构信号」。

## 4. 训练协议（充分训练，杜绝短 POC 判定）

- **数据**：SceneFlow 主训 + 多域（KITTI/Middlebury/ETH3D/Booster）零样本评测；
- **充分收敛**：完整 epoch、warmup + cosine 衰减、与 WAFT 基线同协议；
- **统计严谨**：≥3 种子 mean±std + 配对 t 检验；
- **单因子消融**：关闭 ①/②/③ 某一分支（`gate` 零初始化保证其余等价），逐路归因；
- **明确不做**：以短训练合成数据 EPE 判定模块有效/无效（无效证据，已废止）。

## 5. 创新性说明（诚实：迁移 + 差异化主张）

- **不主张单模块原创**：三路分支 + 融合 + 门控均来自已发表文献（§3）。
- **主张的是「迁移 + 补充式缝合」**：把 WAVE 的 GWCE 从 **ConvGRU 更新器**迁移到 **WAFT 的
  ViT 迭代**上，并以「**旁路门控残差（补充）**」而非「替换更新器输入」的方式缝合——
  这使得代价体融合与 WAFT 自回归**解耦**，可加载预训练 WAFT。
- **与 WAVE-Stereo（prior work）的差异化**：
  1. 更新器：WAVE 用 ConvGRU；本设计保留 WAFT 的 **ViT 迭代**；
  2. 缝合方式：WAVE 在 ConvGRU 输入处替换/扩展为三路；本设计是**旁路门控残差（补充）**，
     原始 warp 迭代一字不动；
  3. 全局上下文：WAVE 用 PGCP（1/32 周期性 ViT）；本设计保留 WAFT 原有迭代，全局上下文
     作为**可选扩展**（GREAT-Stereo / PGCP 均可接入旁路）。
- **定位**：WAVE-Stereo 的**差异化变体 / 迁移消融研究**，而非独立新范式。核心差异需在全量
  训练下逐点对照验证（见 `docs/COMPARISON_PLAN.md`）。
