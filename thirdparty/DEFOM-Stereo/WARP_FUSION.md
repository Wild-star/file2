# DEFOM-Stereo + Warp Fusion（warp 对齐分支）

以 **DEFOM-Stereo（CVPR 2025）为基线**，把 WAFT 的 **warp 残差对齐思路**融入其
Delta Update 的 motion encoder。本目录 vendor 自上游开源实现，仅做补充式改动，
未替换基线任何核心结构。

## 来源

- 论文：DEFOM-Stereo: Depth Foundation Model Based Stereo Matching（CVPR 2025）
- arXiv：2501.09466
- 上游代码：https://github.com/antigravity-tech/DEFOM-Stereo
- 许可证：见 `LICENSE`

## 设计定位（诚实说明）

「warp + correlation 互补」的核心洞察是 **WAVE-Stereo** 先提出的（prior work）。
本实现的差异点在**基线**：WAVE 用 LightStereo 轻量聚合；这里用 **DEFOM-Stereo**
（深度基础模型 Depth Anything V2 先验 + Scale Update + Delta Update），是更强的
SOTA 基线。因此本工作定位为 **DEFOM-Stereo 的 warp 融合变体 / 消融研究**，不是
新范式。

## 接入点

DEFOM-Stereo 的 Delta Update 每轮从相关体金字塔检索 `corr`，与 `disp` 一起送
`BasicMotionEncoder`（Eqn 3 的 motion encoder）：

```
原版:  x_n = [Encoder_c(corr), Encoder_d(disp)]   → ConvGRU
改动:  + Encoder_w(cat[fmap1, disp_warp(fmap2, disp)])  （新增 warp 对齐分支）
```

warp 对齐分支用当前视差 `disp` 把右图匹配特征 `fmap2` 向左对齐到左图坐标系，
与 `fmap1` 拼接后编码，观察对齐残差——即 WAFT 的「warp 残差对齐」思路，与
WAVE GWCE 的 cross-view warping 分支同构。

## 改动清单（4 个文件，均为补充式）

| 文件 | 改动 |
|---|---|
| `core/utils/utils.py` | 新增 `disp_warp(fmap, disp)`：水平方向按视差 warp 右图特征 |
| `core/update.py` | `BasicMotionEncoder` 新增 warp 分支（`convw1/convw2` 正常初始化 + `convw3` **零初始化**残差注入）；`BasicMultiUpdateBlock`（Delta Update）接线传 `fmap1/fmap2`，`use_warp=True`；`ScaleBasicMultiUpdateBlock` 保持 `use_warp=False`（不动） |
| `core/defom_stereo.py` | Delta Update 调用处传 `fmap1, fmap2` |
| `train_stereo.py` | 新增 `--use_warp` / `--no-use_warp` 消融开关 |

## 手术安全（零初始化）

`convw3` 零初始化 → warp 分支初始输出恒为 0。因此：

- **加载上游预训练 DEFOM-Stereo 权重后，行为严格等价原版**（bit-identical 前向）；
- 训练时 `convw3` 先学（梯度非零），`convw1/convw2` 后学（梯度被零初始化阻断），
  分阶段激活，避免初期干扰基线收敛。

## 消融方法

```bash
# 基线（关 warp，等价上游 DEFOM-Stereo）
python train_stereo.py --no-use_warp ...

# 融合（开 warp，本实现）
python train_stereo.py --use_warp ...
```

## 版本

默认骨干 `--dinov2_encoder vits`（Depth Anything V2 的 ViT-Small），即 ViT-S 版本。
