# -*- coding: utf-8 -*-
"""验证 LoRA 里 r 和 alpha 各自干什么：
   ① r 决定 ΔW 的最大秩（容量）
   ② ΔW 的幅度随 √r 增长 → 所以需要 alpha/r（或 rsLoRA 的 alpha/√r）归一化
"""
import numpy as np

d_in = d_out = 256                       # 用 256×256 的线性层做演示
rng = np.random.default_rng(0)

print("① r 决定容量的上限：rank(B·A) 最多是 r")
print(f"{'r':>4}{'参数数 r(d_in+d_out)':>24}{'rank(B·A)':>12}{'相对原始参数':>14}")
base_params = d_in * d_out
for r in (1, 4, 16, 64):
    A = rng.normal(0, 0.01, (r, d_in))
    B = rng.normal(0, 0.01, (d_out, r))
    dW = B @ A
    print(f"{r:>4}{r*(d_in+d_out):>24}{np.linalg.matrix_rank(dW):>12}"
          f"{r*(d_in+d_out)/base_params*100:>13.2f}%")

print("\n② ΔW 的幅度随 √r 增长（A、B 同分布初始化，模拟训练开始后 B 已非零）")
print(f"{'r':>4}{'‖BA‖_F':>12}{'/√r':>10}{'α/r 缩放后(α=32)':>20}{'α/√r 缩放后':>16}")
for r in (4, 16, 64, 256):
    norms = []
    for _ in range(20):                  # 多次采样取平均
        A = rng.normal(0, 0.01, (r, d_in))
        B = rng.normal(0, 0.01, (d_out, r))
        norms.append(np.linalg.norm(B @ A))
    n = np.mean(norms)
    print(f"{r:>4}{n:>12.4f}{n/np.sqrt(r):>10.5f}{n*32/r:>20.5f}{n*32/np.sqrt(r):>16.5f}")

print("\n读法：")
print("  · ‖BA‖_F 与 √r 成正比 → 同样初始化下 r 越大，LoRA 分支的输出越'响'")
print("  · 用 α/r 缩小时，r 越大有效幅度越小（∝1/√r）→ 大 rank 会学不动")
print("  · 用 α/√r 缩放时，幅度与 r 无关 → 这就是 rsLoRA 的动机")
print("  · r=16 时 α/r = 32/16 = 2 → LoRA 分支的输出被放大 2 倍后加到原输出上")
