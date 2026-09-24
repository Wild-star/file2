# FusionWarp-Stereo 使用文档

在 WAFT-Stereo（warping-only、无代价体）基线上，缝合三个可消融模块，形成
「FusionWarp-Stereo」。本文档覆盖环境配置、模型结构、配置项、自检、训练与消融实验。

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

> 可选：`requirements.txt` 里的 `xformers` 已注释，未装不影响训练（DAv2 编码器在无 xformers 时自动降级）。

---

## 2. 模型结构：缝合了哪些模块

基线 `algorithms/waft.py::WAFT` 的迭代更新输入为 `cat[fmap1, warped_fmap2, net, disp]`，
**无任何显式匹配信号**。本次缝合新增 `model/fusion.py`，含四个模块：

| 模块 | 对应方向 | 作用 | 零初始化 |
|---|---|---|---|
| `MatchingBranch` | 方向3（编码器根因） | 从原始图像提「匹配友好」特征（1/4 分辨率，~34K 参数） | — |
| `SparseCorrAnchor` | 方向1（匹配证据） | 在 `disp±R` 内做 group-wise **无聚合**相关，注入迭代 | ✅ 输出恒 0 |
| `GEVCostAnchor` | 方向1 对照 | 窄带代价体 + **可分离 3D 聚合**（对抗性消融对照） | ✅ feat 恒 0 |
| `GlobalMatcher` | 方向2（初始视差） | 1/8 交叉注意力 + 全范围相关 soft-argmax，**直接回归**初始视差 | —（替换起点，非注入） |

**注入位置**（`algorithms/waft.py::forward`）：
1. 编码器后：`m1, m2 = matching_branch(image1/image2)`（1/4 分辨率）；
2. 迭代内：`net = delta_proj(...)` 之后，`net = net + anchor`（anchor 上采样回 1/2）；
3. 初始视差：`USE_GLOBAL_INIT=True` 时 `disp = global_matcher(fmap1, fmap2)` 替换 bins 分类。

**手术安全**：`FUSION.ENABLED=True` 且 `USE_GLOBAL_INIT=False` 时，anchor 零初始化输出
恒为 0 → 前向与原始 WAFT 比特级等价，可从预训练 WAFT 权重直接加载继续训练。

---

## 3. 配置项说明（`WAFT.FUSION` 段）

在 `bridgedepth/config/default.py` 已注册，yaml 里覆盖即可：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `ENABLED` | `False` | 总开关；`False` 时完全等价原版 |
| `USE_ANCHOR` | `True` | 是否注入相关锚（方向1） |
| `USE_GLOBAL_INIT` | `False` | 是否用 GlobalMatcher 替换 bins 初始视差（方向2） |
| `ANCHOR_KIND` | `"corr"` | `"corr"`（无聚合，推荐）/ `"gev"`（3D 聚合，消融对照） |
| `MATCH_CH` | `32` | 匹配分支通道数（输出 1/4 分辨率） |
| `CORR_RADIUS` | `4` | SparseCorrAnchor 窄带半径 `±R` |
| `CORR_GROUPS` | `8` | group-wise 相关的分组数（须整除 `MATCH_CH`） |
| `GEV_AGG_KIND` | `"sep3d"` | gev 的聚合方式：`"sep3d"` / `"full3d"` |
| `GEV_K` / `GEV_R` | `9` / `8` | gev 的候选数 / 半径 |
| `GM_HEADS` | `4` | GlobalMatcher 交叉注意力 head 数 |
| `GM_NDISP` | `null` | 全范围相关候选数；`null` = `MAX_DISP // 8` |

示例配置见 `configs/SynLarge/DAv2S-4-fusion.yaml`（基于 DAv2S-4 增加 `FUSION` 段）。

---

## 4. 自检（先确认缝合无误）

```bash
# 模块级自检（快，验证零初始化 + 形状 + 梯度）
python test_fusion.py

# 完整 WAFT 前向+反向（DAv2-Small，128×160 小输入，验证梯度流到匹配分支）
python test_fusion.py --full

# 训练冒烟（最少随机数据，跑 N 步 forward/loss/backward/step）
python train_smoke.py                          # corr 锚，5 步
STEPS=10 ANCHOR=gev python train_smoke.py      # gev 锚
STEPS=5 GLOBAL_INIT=1 python train_smoke.py    # 加 GlobalMatcher
```

**已验证结果**：三条路径（corr / gev / global_init）训练循环均跑通、loss 下降；
`SparseCorrAnchor` / `GEVCostAnchor` 零初始化实测输出恒 0（手术安全确认）。

---

## 5. 训练

```bash
# 主方案：无聚合相关锚
python main.py --num-gpus 1 --config-file configs/SynLarge/DAv2S-4-fusion.yaml

# 多卡
python main.py --num-gpus 8 --config-file configs/SynLarge/DAv2S-4-fusion.yaml

# 从预训练 WAFT 权重继续训练（手术安全：anchor 零初始化，起点等价原版）
python main.py --num-gpus 8 --config-file configs/SynLarge/DAv2S-4-fusion.yaml \
    SOLVER.RESUME ckpts/SynLarge/DAv2S-4.pth
```

> **前提**：真实训练前需把 DAv2-Small 权重放到 `depth-anything-ckpts/depth_anything_v2_vits.pth`，
> 否则编码器从随机权重初始化（自检模式可接受，正式训练不可）。

---

## 6. 消融实验设计（回答 §9.3 的三个命题）

| 配置（改 yaml 的 `FUSION` 段） | 验证目标 |
|---|---|
| `ENABLED=False` | 基线 WAFT（无匹配证据） |
| `ENABLED=True, USE_ANCHOR=True, ANCHOR_KIND=corr` | **主方案**：无聚合相关锚 |
| `ENABLED=True, USE_ANCHOR=True, ANCHOR_KIND=gev` | 命题①：无聚合 vs 3D 聚合（对抗 WAVE 的几何编码体积） |
| `ENABLED=True, USE_ANCHOR=False` | 锚的必要性（方向1 消融） |
| `ENABLED=True, USE_GLOBAL_INIT=True` | 命题②/方向2：直接回归 vs bins 分类初始视差 |
| corr + GlobalMatcher 全开 | 三模块叠加的完整 FusionWarp |

---

## 7. 已知问题与注意事项

1. **显存**：本机 RTX 5060 Laptop 8GB，仅够自检与小批调试；DAv2-Small + 480×640 crop
   的单卡训练可能 OOM，全量训练建议放多卡服务器（原论文 DAv2-Large 用 8×A100）。
2. **`grid_sample` align_corners 警告**：来自原始 `model/utils.py::disp_warp`（未指定
   `align_corners`），是基线既有行为，未改动，避免破坏预训练权重兼容性。
3. **GlobalMatcher 从零训练**：交叉注意力/全范围相关无预训练权重，收敛速度需观察；
   若拖慢整体，可考虑加辅助监督或减小 `GM_NDISP`。
4. **xformers 未装**：不影响正确性，仅损失 DAv2 的注意力加速。

---

## 8. 文件清单（本次缝合）

| 文件 | 说明 |
|---|---|
| `model/fusion.py` | 四个融合模块 |
| `algorithms/waft.py` | WAFT 三处注入 |
| `bridgedepth/config/default.py` | `WAFT.FUSION` 配置段 |
| `configs/SynLarge/DAv2S-4-fusion.yaml` | 可训练配置示例 |
| `test_fusion.py` | 模块级 + 完整自检 |
| `train_smoke.py` | 训练冒烟测试 |
| `docs/paper_design_fusionwarp.md` | 论文设计（§9 已修订为跨象限差异化定位） |
