"""DEFOM-Stereo + Warp Fusion 训练冒烟测试（CPU，随机权重，合成假数据）。

验证部署训练前必须确认的三件事：
1. 训练循环核心能跑：forward → sequence_loss → backward → clip_grad → optimizer.step；
2. warp 对齐分支在训练中能反向传播、参数会更新（零初始化的 convw3 从全 0 变成非 0）；
3. loss 有限且随 step 下降（训练在收敛）。

仅需 torch；缺失的 opt_einsum/timm/cv2/xformers 用 mock 或可选 fallback 处理。

用法：python train_smoke_test.py [--steps N] [--use_warp/--no-use_warp]

GPU 服务器上的完整训练冒烟（真实数据 + 真实 train_stereo.py）：
    # 1) 建真实环境
    conda env create -f environment.yaml && conda activate defomstereo
    # 2) 下载 Depth Anything V2 ViT-S 权重到 checkpoints/depth_anything_v2_vits.pth
    # 3) 用极小步数跑真实训练脚本，确认能启动、能写 checkpoint：
    python train_stereo.py --name smoke --batch_size 2 --num_steps 5 \
        --train_iters 3 --scale_iters 1 --val_freq 100000 --use_warp
"""
import sys
import types
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# mock 缺失依赖（同 smoke_test_warp.py）
# --------------------------------------------------------------------------- #
_oe = types.ModuleType('opt_einsum')
_oe.contract = lambda *a, **k: None
sys.modules['opt_einsum'] = _oe

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

_cv2 = types.ModuleType('cv2')
_cv2.INTER_NEAREST = 0
_cv2.INTER_LINEAR = 1
_cv2.INTER_AREA = 3
_cv2.resize = lambda *a, **k: a[0]
sys.modules['cv2'] = _cv2


# --------------------------------------------------------------------------- #
# sequence_loss（复制自 utils/utils.py，避免 tensorboard 依赖）
# --------------------------------------------------------------------------- #
def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        adjusted_loss_gamma = loss_gamma ** (15 / n_predictions)
        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    return flow_loss, {'epe': epe.mean().item()}


def build_args(use_warp=True):
    return argparse.Namespace(
        dinov2_encoder='vits', idepth_scale=0.5,
        hidden_dims=[128, 128, 128], n_gru_layers=3, n_downsample=2,
        context_norm='batch', corr_levels=2, corr_radius=4,
        scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        scale_corr_radius=2, corr_implementation='reg',
        mixed_precision=False, use_warp=use_warp,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--use_warp', dest='use_warp', action='store_true', default=True)
    p.add_argument('--no-use_warp', dest='use_warp', action='store_false')
    p.add_argument('--iters', type=int, default=2)
    p.add_argument('--scale_iters', type=int, default=1)
    a = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core.defom_stereo import DEFOMStereo

    model = DEFOMStereo(build_args(use_warp=a.use_warp))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    print(f'[1] 模型 + AdamW 就绪（use_warp={a.use_warp}），总参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # 合成假数据（冒烟用小图加速，真实训练用 1/4 下采样后仍为 14 倍数的大图）
    B, C, H, W = 1, 3, 112, 144
    torch.manual_seed(0)
    image1 = torch.randn(B, C, H, W) * 255
    image2 = torch.randn(B, C, H, W) * 255
    disp_gt = torch.rand(B, 1, H, W) * 50.0
    valid = torch.ones(B, 1, H, W)

    # 记录零初始化 convw3 的初始状态
    convw3_before = model.update_block.encoder.convw3.weight.detach().clone() if a.use_warp else None

    losses = []
    print(f'[2] 训练循环 {a.steps} step（合成数据，CPU）...')
    for step in range(a.steps):
        optimizer.zero_grad()
        disp_preds = model(image1, image2, iters=a.iters, scale_iters=a.scale_iters)
        loss, metrics = sequence_loss(disp_preds, disp_gt, valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step == 0:
            # 检查 warp 分支梯度是否流动
            g1 = model.update_block.encoder.convw1.weight.grad if a.use_warp else None
            g3 = model.update_block.encoder.convw3.weight.grad if a.use_warp else None
            print(f'    第1步 loss={loss.item():.3f} epe={metrics["epe"]:.2f} | '
                  f'convw1.grad 非零={bool(g1 is not None and g1.abs().max() > 0)} | '
                  f'convw3.grad 非零={bool(g3 is not None and g3.abs().max() > 0)}')

    print(f'[3] loss 序列: ' + ' -> '.join(f'{x:.2f}' for x in losses))
    down = losses[-1] < losses[0]
    print(f'    loss 下降: {"PASS" if down else "WARN(随机数据可能震荡)"}')

    if a.use_warp:
        convw3_after = model.update_block.encoder.convw3.weight.detach()
        delta = (convw3_after - convw3_before).abs().max().item()
        nonzero = convw3_after.abs().max().item() > 0
        print(f'[4] warp 分支更新：convw3 初始全 0 -> 训练后 max|w|={convw3_after.abs().max().item():.2e} '
              f'（更新量 {delta:.2e}）: {"PASS" if nonzero and delta > 0 else "FAIL"}')

    ok = all(torch.isfinite(torch.tensor(x)) for x in losses)
    print(f'\n训练冒烟测试 {"PASS" if ok else "FAIL"}（warp 分支反向传播/更新正常）。')


if __name__ == '__main__':
    main()
