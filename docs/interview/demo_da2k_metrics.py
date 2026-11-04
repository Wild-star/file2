# -*- coding: utf-8 -*-
"""演示两个东西的计算：① 训练损失（scale-shift invariant）② DA-2K 的指标（成对准确率）
并对比"旧指标为什么骗人"。"""
import numpy as np

# ---------------- ① 训练损失：scale-shift invariant (SSI) ----------------
def ssi_normalize(d):
    """把深度图归一化：中心化到中位数为 0，再除以平均绝对偏差。
    关键：预测和 GT 两边都要做，尺度/平移才被消除。"""
    t = np.median(d)                      # 平移量
    s = np.abs(d - t).mean()              # 尺度量（平均绝对偏差）
    return (d - t) / s

def ssi_loss(pred, gt):
    return np.abs(ssi_normalize(pred) - ssi_normalize(gt)).mean()

# ---------------- 旧指标：AbsRel / RMSE / δ ----------------
def absrel(pred, gt):
    return np.abs(pred - gt).mean() / gt.mean()

def rmse(pred, gt):
    return np.sqrt(((pred - gt) ** 2).mean())

def delta1(pred, gt):
    r = np.maximum(pred / gt, gt / pred)
    return (r < 1.25).mean()

# ---------------- ② DA-2K 指标：成对深度准确率 ----------------
def pairwise_accuracy(pred, pairs, labels):
    """pairs: [(i,j), ...] 像素下标；labels: 1 表示 j 更近（深度更小），0 表示 i 更近"""
    correct = 0
    for (i, j), lab in zip(pairs, labels):
        pred_label = 1 if pred[j] < pred[i] else 0      # 深度小 = 更近
        correct += (pred_label == lab)
    return correct / len(pairs)

# ---------------- 造数据做三组实验 ----------------
rng = np.random.default_rng(0)
H = W = 64
# 构造一张"有层次"的深度图：左边近、右边远，中间有个物体（细节）
yy, xx = np.mgrid[0:H, 0:W]
gt = 1.0 + 0.02 * xx + 0.01 * yy                      # 基础斜面 1~3 米
gt[20:30, 20:30] += -0.8                              # 一个"近处物体"，产生细节边界

# 采样成对标注（DA-2K 的做法：随机像素对 + 边界附近加密）
def sample_pairs(n=4000, n_edge=3000, seed=0):
    r = np.random.default_rng(seed)
    pairs, labels = [], []
    for _ in range(n):                                # 普通随机对
        i, j = r.integers(0, H * W, 2)
        if i == j: continue
        pairs.append((i, j)); labels.append(1 if gt.ravel()[j] < gt.ravel()[i] else 0)
    # 边界附近的对（细节区）：物体角落与周围像素
    for _ in range(n_edge):
        i = r.integers(20 * W + 20, 30 * W + 30)       # 物体内部
        j = r.integers(0, H * W)                       # 任意点
        pairs.append((i, j)); labels.append(1 if gt.ravel()[j] < gt.ravel()[i] else 0)
    return pairs, labels

pairs, labels = sample_pairs()

cases = {
    "A 完美预测": gt.copy(),
    "B 全局尺度错2倍+平移3": gt * 2 + 3,          # 只差一个仿射变换
    "C 细节幻觉(物体区域反了)": None,
}
c = gt.copy()
c[20:30, 20:30] = gt.max() - c[20:30, 20:30]          # 把物体区的相对关系搞反
cases["C 细节幻觉(物体区域反了)"] = c

print(f"{'情况':<24}{'AbsRel':>9}{'RMSE':>9}{'δ<1.25':>9}{'SSI损失':>10}{'成对准确率':>12}")
print("-" * 74)
for name, pred in cases.items():
    p = pred.ravel()
    print(f"{name:<24}{absrel(p, gt.ravel()):>9.3f}{rmse(p, gt.ravel()):>9.3f}"
          f"{delta1(p, gt.ravel()):>9.3f}{ssi_loss(p, gt.ravel()):>10.4f}"
          f"{pairwise_accuracy(p, pairs, labels):>12.3f}")

# ---- 分开统计：随机对 vs 边界/细节对（说明 DA-2K 为什么要在挑战区域加密标注）----
print()
print("把成对准确率拆开看（C 情况 = 细节幻觉）：")
c_pred = cases["C 细节幻觉(物体区域反了)"].ravel()
n_rand = 4000
acc_rand = pairwise_accuracy(c_pred, pairs[:n_rand], labels[:n_rand])
acc_edge = pairwise_accuracy(c_pred, pairs[n_rand:], labels[n_rand:])
print(f"   随机像素对  : {acc_rand:.3f}   ← 看起来很好，因为大片区域是对的")
print(f"   边界/细节对 : {acc_edge:.3f}   ← 细节幻觉在这里暴露")
print(f"   合计        : {pairwise_accuracy(c_pred, pairs, labels):.3f}")
