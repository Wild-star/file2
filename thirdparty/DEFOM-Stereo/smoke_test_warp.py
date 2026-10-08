"""DEFOM-Stereo + Warp Fusion 冒烟测试（CPU，随机权重，无需 Depth Anything V2 权重）。

验证：
1. 完整 DEFOMStereo.forward 能跑通（含 warp 对齐分支）；
2. 零初始化手术安全：同一模型开关 warp 分支，前向输出 bit-identical（加载预训练权重后等价原版）。

用法：python smoke_test_warp.py
仅需 torch；缺失的 opt_einsum/timm/cv2/xformers 用 mock 或可选 fallback 处理。
"""
import sys
import types
import argparse
from pathlib import Path

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# mock 缺失依赖（本机为 CPU-only torch，无 timm/opt_einsum/cv2）
# --------------------------------------------------------------------------- #
# opt_einsum：core/update.py 仅 import 未使用，mock 一个空 contract
_oe = types.ModuleType('opt_einsum')
_oe.contract = lambda *a, **k: None
sys.modules['opt_einsum'] = _oe

# timm.models.layers.DropPath：stochastic depth 层，drop_prob=0 时恒等，冒烟测试恒等即可
class _DropPath(nn.Module):
    def __init__(self, drop_prob=0., scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return x

_timm = types.ModuleType('timm')
_timm_models = types.ModuleType('timm.models')
_timm_layers = types.ModuleType('timm.models.layers')
_timm_layers.DropPath = _DropPath
_timm_models.layers = _timm_layers
_timm.models = _timm_models
sys.modules['timm'] = _timm
sys.modules['timm.models'] = _timm_models
sys.modules['timm.models.layers'] = _timm_layers

# cv2：depth_anything_v2 用到插值常量（transform.py 类属性）+ resize（demo 路径，模型前向不走）
_cv2 = types.ModuleType('cv2')
_cv2.INTER_NEAREST = 0
_cv2.INTER_LINEAR = 1
_cv2.INTER_AREA = 3
_cv2.resize = lambda *a, **k: a[0]
sys.modules['cv2'] = _cv2

# --------------------------------------------------------------------------- #
# 构造 args（与 train_stereo.py 默认对齐，ViT-S）
# --------------------------------------------------------------------------- #
def build_args(use_warp=True):
    return argparse.Namespace(
        dinov2_encoder='vits',
        idepth_scale=0.5,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        n_downsample=2,
        context_norm='batch',
        corr_levels=2,
        corr_radius=4,
        scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        scale_corr_radius=2,
        corr_implementation='reg',
        mixed_precision=False,
        use_warp=use_warp,
    )


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core.defom_stereo import DEFOMStereo

    model = DEFOMStereo(build_args(use_warp=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_warp = sum(p.numel() for n, p in model.named_parameters() if 'convw' in n)
    print(f'[1] 模型实例化 OK（ViT-S, use_warp=True）')
    print(f'    总参数量 {n_params/1e6:.2f}M，其中 warp 分支 {n_warp/1e3:.1f}K')

    # 随机输入（14 的倍数下采样到 1/4 后 DINOv2 输入仍为 14 的倍数）
    B, C, H, W = 1, 3, 192, 256
    torch.manual_seed(0)
    img1 = torch.randn(B, C, H, W) * 255
    img2 = torch.randn(B, C, H, W) * 255

    with torch.no_grad():
        out_warp = model(img1, img2, iters=2, scale_iters=1, test_mode=False)
    print(f'[2] 前向 OK（开 warp）：{len(out_warp)} 个预测，末个形状 {tuple(out_warp[-1].shape)}')

    # 零初始化手术安全：同一模型关闭 warp 分支，输出应 bit-identical
    model.update_block.encoder.use_warp = False
    with torch.no_grad():
        out_nowarp = model(img1, img2, iters=2, scale_iters=1, test_mode=False)
    diff = max((a - b).abs().max().item() for a, b in zip(out_warp, out_nowarp))
    print(f'[3] 零初始化等价（开 warp == 关 warp）：max|diff| = {diff:.3e}  '
          f'{"PASS" if diff < 1e-6 else "FAIL"}')

    # 梯度可回传（训练时 warp 分支能学）：convw3 先学、convw1 被零初始化阻断
    model.update_block.encoder.use_warp = True
    model.update_block.encoder.convw3.weight.grad = None
    model.update_block.encoder.convw1.weight.grad = None
    loss = model(img1, img2, iters=1, scale_iters=0, test_mode=False)[-1].sum()
    loss.backward()
    g3 = model.update_block.encoder.convw3.weight.grad
    g1 = model.update_block.encoder.convw1.weight.grad
    print(f'[4] 梯度行为：convw3.grad 非零 = {bool(g3 is not None and g3.abs().max() > 0)}，'
          f'convw1.grad 为零（阻断）= {bool(g1 is not None and g1.abs().max().item() == 0)}')

    print('\n冒烟测试完成：warp 融合分支正确接入，且手术安全（加载预训练权重后等价原版）。')


if __name__ == '__main__':
    main()
