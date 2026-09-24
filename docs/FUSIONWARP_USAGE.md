# FusionWarp-Stereo 使用文档

在 WAFT-Stereo（warping-only、无代价体）基线上，做**深度模块融合**：把「可学习匹配、
门控融合、token 稀疏」等深度学习模块缝合进 WAFT 的迭代更新。本文档覆盖环境配置、
模型结构、配置项、自检、训练与消融实验。

> **差异化定位**（相对 WAVE-Stereo 的跨象限差异化，详见 `docs/paper_design_fusionwarp.md` §9）：
> 不在「首次统一相关+warp」上争原创，而是回答三个可证伪命题——
> ① 无聚合相关 > 几何编码体积聚合；② ViT 原生全局 vs ConvGRU+周期 PGCP；③ VFM 先验 + 最小匹配分支。

---

## 1. 环境配置（已在本机验证通过）

- 包管理器：**uv**（`~/.local/bin/uv`，版本 ≥ 0.12）
- Python：**3.12**
- PyTorch：**2.8.0+cu128**（CUDA 12.8，GPU 可用）

```bash
cd ~/桌面/file2

# 1. 创建虚拟环境
uv venv .venv --python 3.12
source .venv/bin/activate          # 或直接用 .venv/bin/python

# 2. 安装 PyTorch（CUDA 12.8 版，务必用官方 cu128 index）
uv pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. 安装其余依赖
uv pip install -r requirements.txt

# 4. 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出：2.8.0+cu128 True
```

---

## 2. 模型结构：缝合了哪些模块

基线 `algorithms/waft.py::WAFT` 的迭代更新为 `cat[fmap1, warped_fmap2, net, disp]`，
**无任何显式匹配信号**。本次缝合 `model/fusion.py` + `model/iterative/vit.py`，含以下模块：

| 模块 | 本质 | 参数量 | 零初始化 |
|---|---|---|---|
| `MatchingBranch` | 可学习 CNN 提匹配特征（1/4 分辨率） | 30.2K | — |
| `SparseCorrAnchor` | 手工相关 + 可学习 3×3 投影 | 31.2K | ✅ 输出恒 0 |
| `GEVCostAnchor` | 可学习可分离 3D 聚合（消融对照） | 4.9K | ✅ feat 恒 0 |
| `GlobalMatcher` | **可学习 QKV 交叉注意力** + 全范围相关 soft-argmax | 21.3K | —（替换起点） |
| `GatedFusion` | **可学习门控融合**（局部锚 ⊕ 全局上下文） | 0.9K | ✅ out_scale=0 |
| `TokenSparseVitIter` | **可学习 token 稀疏**（saliency 门控更新） | +0.4K | ✅ gate=1 |

**三条注入路径**（`algorithms/waft.py::forward`）：
1. 编码器后：`m1, m2 = matching_branch(image1/image2)`（1/4 分辨率）；
2. 初始视差：`USE_GLOBAL_INIT=True` 时 `d_gm, g_feat = global_matcher(fmap1, fmap2)`，
   `d_gm` 替换 bins 分类起点；
3. 迭代内：`anchor = corr_anchor(m1, m2, disp_q)` → 上采样回 1/2 →
   `USE_GATED_FUSION=True` 时 `fused = gated_fusion(anchor, g_feat)`，否则 `net += anchor`。

**迭代解码器**：`DELTA_ITER.TYPE='vit'`（原 VitIter）或 `'vit_sparse'`（TokenSparseVitIter）。

**手术安全**：`USE_GLOBAL_INIT=False` 时，anchor/fused 零初始化输出恒 0 → 前向与原始
WAFT 比特级等价，可从预训练 WAFT 权重直接加载继续训练。

### 2.1 深度模块融合（路线 B）vs 相关算子（路线 A）

- **路线 A（轻量算子注入）**：只有 MatchingBranch + 手工相关 + 小投影，参数量 ~61K，
  「深度学习含量」低，核心匹配是手工点积。
- **路线 B（深度模块融合）**：新增 `GlobalMatcher` 可学习 QKV 交叉注意力、
  `GatedFusion` 门控融合、`TokenSparseVitIter` 可学习 token 稀疏，参数量 ~84K，
  匹配/融合/稀疏都有可学习组件，更符合「模块融合」定位。

---

## 3. 配置项说明

`WAFT.FUSION` 段（`bridgedepth/config/default.py` 已注册）：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `ENABLED` | `False` | 总开关；`False` 时完全等价原版 |
| `USE_ANCHOR` | `True` | 是否注入相关锚（方向1） |
| `USE_GLOBAL_INIT` | `False` | 是否用 GlobalMatcher 替换 bins 初始视差（方向2） |
| `USE_GATED_FUSION` | `False` | 是否用 GatedFusion 门控融合（隐含 USE_ANCHOR，需 USE_GLOBAL_INIT 提供 g_feat） |
| `ANCHOR_KIND` | `"corr"` | `"corr"`（无聚合）/ `"gev"`（3D 聚合，消融对照） |
| `MATCH_CH` | `32` | 匹配分支通道数（输出 1/4 分辨率） |
| `CORR_RADIUS` | `4` | 相关锚窄带半径 `±R` |
| `CORR_GROUPS` | `8` | group-wise 相关分组数（须整除 `MATCH_CH`） |
| `GEV_AGG_KIND` | `"sep3d"` | gev 聚合：`"sep3d"` / `"full3d"` |
| `GEV_K` / `GEV_R` | `9` / `8` | gev 候选数 / 半径 |
| `GM_HEADS` | `4` | GlobalMatcher 交叉注意力 head 数 |
| `GM_NDISP` | `null` | 全范围相关候选数；`null` = `MAX_DISP // 8` |

`WAFT.ITERATIVE_MODULE.DELTA_ITER.TYPE`：`"vit"`（VitIter）/ `"vit_sparse"`（TokenSparseVitIter）。

---

## 4. 自检（先确认缝合无误）

```bash
python test_fusion.py                          # 模块级（快，含 QKV/GatedFusion 零初始化验证）
python test_fusion.py --full                   # 完整 WAFT 前向+反向（corr/global_init/gated_fusion）

python train_smoke.py                          # corr 锚，5 步
STEPS=3 GLOBAL_INIT=1 GATED_FUSION=1 python train_smoke.py   # 完整深度融合
STEPS=3 TOKEN_SPARSE=1 python train_smoke.py                # token 稀疏
```

**已验证**：全部路径训练循环跑通、loss 下降；`SparseCorrAnchor`/`GEVCostAnchor`/
`GatedFusion` 零初始化实测输出恒 0（手术安全确认）；`TokenSparseVitIter` saliency 零
初始化 → gate=1（等价原版 VitIter）。

---

## 5. 训练

```bash
# 路线 B 完整深度融合（corr 锚 + GlobalMatcher 初始视差 + GatedFusion 门控融合）
python main.py --num-gpus 8 --config-file configs/SynLarge/DAv2S-4-fusion.yaml \
    WAFT.FUSION.USE_GLOBAL_INIT True WAFT.FUSION.USE_GATED_FUSION True

# 加 token 稀疏解码器
python main.py --num-gpus 8 --config-file configs/SynLarge/DAv2S-4-fusion.yaml \
    WAFT.ITERATIVE_MODULE.DELTA_ITER.TYPE vit_sparse

# 从预训练 WAFT 权重继续训练（手术安全：零初始化，起点等价原版）
python main.py --num-gpus 8 --config-file configs/SynLarge/DAv2S-4-fusion.yaml \
    SOLVER.RESUME ckpts/SynLarge/DAv2S-4.pth
```

> **前提**：真实训练前需把 DAv2-Small 权重放到 `depth-anything-ckpts/depth_anything_v2_vits.pth`。

---

## 6. 消融实验设计（回答 §9.3 的三个命题）

| 配置 | 验证目标 |
|---|---|
| `ENABLED=False` | 基线 WAFT（无匹配证据） |
| `USE_ANCHOR=True, ANCHOR_KIND=corr` | 主方案：无聚合相关锚 |
| `USE_ANCHOR=True, ANCHOR_KIND=gev` | 命题①：无聚合 vs 3D 聚合（对抗 WAVE 几何编码体积） |
| `USE_ANCHOR=False` | 锚必要性（方向1 消融） |
| `USE_GLOBAL_INIT=True` | 命题②/方向2：直接回归 vs bins 分类初始视差 |
| `USE_GLOBAL_INIT=True, USE_GATED_FUSION=True` | 命题②/融合：门控融合全局上下文（vs 直接加锚） |
| 上者 + `DELTA_ITER.TYPE=vit_sparse` | 命题②/token 稀疏：ViT 选择性更新（ConvGRU 无法做） |

---

## 7. 已知问题与注意事项

1. **显存**：本机 RTX 5060 Laptop 8GB，仅够自检与小批调试；全量训练建议放多卡服务器。
2. **`grid_sample` align_corners 警告**：来自原始 `model/utils.py::disp_warp`，是基线既有
   行为，未改动，避免破坏预训练权重兼容性。
3. **GlobalMatcher / GatedFusion / TokenSparse 从零训练**：无预训练权重，收敛速度需观察。
4. **GatedFusion 依赖**：`USE_GATED_FUSION=True` 隐含 `USE_ANCHOR`（需要 anchor 作为输入），
   且需 `USE_GLOBAL_INIT=True` 或至少 GlobalMatcher 运行来提供 g_feat。
5. **xformers 未装**：不影响正确性，仅损失 DAv2 的注意力加速。

---

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `model/fusion.py` | MatchingBranch / SparseCorrAnchor / GEVCostAnchor / GlobalMatcher / GatedFusion |
| `model/iterative/vit.py` | VitIter + TokenSparseVitIter |
| `model/iterative/__init__.py` | 注册 `vit` / `vit_sparse` |
| `algorithms/waft.py` | WAFT 注入（匹配分支 + 锚 + 门控融合 + 初始视差） |
| `bridgedepth/config/default.py` | `WAFT.FUSION` 配置段 |
| `configs/SynLarge/DAv2S-4-fusion.yaml` | 可训练配置示例 |
| `test_fusion.py` | 模块级 + 完整自检 |
| `train_smoke.py` | 训练冒烟测试 |
| `docs/paper_design_fusionwarp.md` | 论文设计（§9 跨象限差异化定位） |
