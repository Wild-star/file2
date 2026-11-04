# -*- coding: utf-8 -*-
"""激活函数的关键性质实测：导数上界、死亡 ReLU、GELU 近似误差、耗时、SwiGLU 参数量"""
import numpy as np, time

# ---------- 定义 ----------
def sigmoid(x): return 1/(1+np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def leaky(x, a=0.01): return np.where(x > 0, x, a*x)
def elu(x, a=1.0): return np.where(x > 0, x, a*(np.exp(np.minimum(x, 0))-1))
def gelu_exact(x):
    from math import erf, sqrt
    return x * 0.5 * (1 + np.vectorize(erf)(x/np.sqrt(2)))
def gelu_tanh(x): return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def silu(x): return x*sigmoid(x)

acts = {"sigmoid": sigmoid, "tanh": tanh, "ReLU": relu, "LeakyReLU": leaky,
        "ELU": elu, "GELU": gelu_exact, "SiLU": silu}

print("① 导数的最大值（决定梯度能不能穿过深网络）")
print(f"{'激活':<10}{'max f0':>10}{'负区间梯度':>12}")
for name, f in acts.items():
    eps = 1e-6
    xs = np.linspace(-8, 8, 4001)
    d = (f(xs+eps) - f(xs-eps)) / (2*eps)
    neg = d[xs < -2].mean()
    print(f"{name:<10}{d.max():>10.4f}{neg:>12.5f}")

print("\n② 死亡 ReLU 演示：一个神经元被推进全负区间后，梯度恒 0、权重永久不更新")
rng = np.random.default_rng(0)
W_true = np.array([0.5, -0.3, 0.8])
X = rng.normal(0, 1, (200, 3))
y = (X @ W_true > 0).astype(float)

def train(act, dact, name):
    """act: 前向；dact: 解析导数（用数值差分容易因间距设置出错，这里用解析式）"""
    w = rng.normal(0, 0.1, 3); b = -10.0      # 关键：bias 人为设成很大的负数
    w0, b0 = w.copy(), b
    for _ in range(30):
        z = X @ w + b
        h = act(z)
        dL_dz = -2*(y - h) * dact(z)
        w -= 0.1 * (X.T @ dL_dz) / len(X)
        b -= 0.1 * dL_dz.mean()
    moved = np.linalg.norm(w - w0) + abs(b - b0)
    print(f"   {name:<12} 训练 30 步后参数总位移 = {moved:.6f}"
          f"   {'← 完全不动 = 已死亡' if moved < 1e-9 else '← 还能恢复'}")

train(relu, lambda z: (z > 0).astype(float), "ReLU")
train(lambda x: leaky(x, 0.01), lambda z: np.where(z > 0, 1.0, 0.01), "LeakyReLU")

print("\n③ GELU 的 tanh 近似与精确值误差")
xs = np.linspace(-6, 6, 13)
err = np.abs(gelu_tanh(xs) - gelu_exact(xs))
print(f"   |x|<=6 范围内最大绝对误差 = {err.max():.5f}（工程上完全可以忽略）")

print("\n④ 向量化耗时（1e6 个元素，GELU 用 tanh 近似版以免 numpy 无原生 erf 的干扰）")
x = rng.normal(0, 1, 1_000_000)
for name, f in (("ReLU", relu), ("SiLU", silu), ("GELU(tanh)", gelu_tanh), ("sigmoid", sigmoid)):
    t0 = time.perf_counter(); [f(x) for _ in range(5)]; dt = (time.perf_counter()-t0)/5*1000
    print(f"   {name:<11}{dt:7.2f} ms")
print("   （数量级示意：ReLU 最省，含 exp/tanh 的都要多几倍；GPU/NPU 上差距更明显）")

print("\n⑤ SwiGLU 为什么要缩小隐藏维（保持参数量与标准 FFN 相当）")
d, F = 4096, 11008          # LLaMA-7B 的 hidden 与 FFN 维度
ffn = 2 * d * F
for name, mult, gates in (("标准 FFN (2 矩阵)", 1.0, 2), ("SwiGLU (3 矩阵)", 8/3/2*2/1, 3)):
    pass
# 标准 FFN 两个矩阵；SwiGLU 三个矩阵 → 每个矩阵用 2/3·F 才能持平
F_swiglu = int(2 * F / 3)
swi = 3 * d * F_swiglu
print(f"   标准 FFN: 2 x 4096 x 11008 = {ffn/1e6:.1f} M")
print(f"   SwiGLU 若用同样 F=11008: 3 x 4096 x 11008 = {3*d*F/1e6:.1f} M（多 50%）")
print(f"   SwiGLU 用 F'=2/3·F={F_swiglu}: 3 x 4096 x {F_swiglu} = {swi/1e6:.1f} M（基本持平）")
