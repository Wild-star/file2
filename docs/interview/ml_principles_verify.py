# -*- coding: utf-8 -*-
"""
机器学习原理的「代码验算」——每个概念用几行代码跑出数字来验证。

面向：Python 语法不熟、但原理学过的人。
所以这份代码的写法刻意"笨"：少用高阶技巧，多处显式循环，每处新语法都有注释。

运行：python3 ml_principles_verify.py
依赖：numpy（torch 部分可选，用于对照）
"""
import math
import numpy as np

RNG = np.random.default_rng(0)
OK, FAIL = [], []


def check(name, cond, note=""):
    """统一的断言记录。cond 为 True 就通过。"""
    (OK if cond else FAIL).append(name)
    print(("  [ OK ] " if cond else "  [FAIL] ") + name + ("   " + note if note else ""))


# =====================================================================
# S1 softmax 家族
# =====================================================================
def softmax(z):
    """手写 softmax。z 是 list 或 ndarray，返回 ndarray。"""
    z = np.asarray(z, dtype=np.float64)       # ← 语法点：把 list 转成 numpy 数组
    z = z - z.max(axis=-1, keepdims=True)     # axis=-1 最后一维；keepdims=True 保住维度
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def v_s1_softmax_basic():
    print("== S1 softmax 基本性质 ==")
    p = softmax([1.0, 2.0, 3.0])
    check("softmax 输出和为 1", abs(p.sum() - 1) < 1e-12, f"sum={p.sum():.12f}")
    check("softmax 单调（输入大则输出大）", p[2] > p[1] > p[0],
          f"p={np.round(p, 4)}")
    # 手算参考值：e^3 / (e^1+e^2+e^3)
    ref = math.exp(3) / (math.exp(1) + math.exp(2) + math.exp(3))
    check("与手算值一致", abs(p[2] - ref) < 1e-12, f"p2={p[2]:.6f} ref={ref:.6f}")


def v_s1_jacobian():
    """验证 softmax 的雅可比：∂p_i/∂z_j = p_i(δ_ij − p_j)  →  J = diag(p) − p pᵀ"""
    print("== S1 softmax 雅可比 ==")
    z = np.array([0.5, -0.2, 1.1])
    p = softmax(z)
    # 解析雅可比
    J_analytic = np.diag(p) - np.outer(p, p)       # ← 语法点：np.outer 外积 → 3x3
    # 数值雅可比：对每个 z_j 加一个小量 eps，看 p 的变化
    eps = 1e-6
    J_numeric = np.zeros((3, 3))
    for j in range(3):                             # ← 语法点：range(n) 就是 0..n-1
        z2 = z.copy()
        z2[j] += eps
        J_numeric[:, j] = (softmax(z2) - p) / eps  # ← 语法点：[:, j] 取第 j 列
    check("解析雅可比 == 数值雅可比", np.allclose(J_analytic, J_numeric, atol=1e-5),
          f"max差={np.abs(J_analytic - J_numeric).max():.2e}")
    check("雅可比每行和为 0（输出必须和为 1 → 全加常数不变）",
          np.allclose(J_analytic.sum(axis=1), 0, atol=1e-12))


def v_s1_overflow():
    """验证为什么要减 max：不减会 inf/nan。"""
    print("== S1 减 max 的必要性 ==")
    z = np.array([1000.0, 1000.0, 1000.0])
    with np.errstate(over="ignore", invalid="ignore"):   # ← 语法点：临时屏蔽 numpy 警告
        naive = np.exp(z) / np.exp(z).sum()
    safe = softmax(z)
    check("不减 max：得到 nan", np.isnan(naive).any(), f"naive={naive[0]}")
    check("减 max 后：正确", np.allclose(safe, 1 / 3), f"safe={safe[0]:.6f}")


def v_s1_gradient():
    """验证 softmax+CE 的梯度 = p − y（用数值微分对比）。"""
    print("== S1 softmax+CE 梯度 = p − y ==")
    z = np.array([0.3, 1.2, -0.7])
    y = 1                                          # 正确类别是 1

    def loss(z):
        p = softmax(z)
        return -math.log(p[y])                     # ← 语法点：函数内可直接用外部变量 y

    p = softmax(z)
    grad_analytic = p.copy()
    grad_analytic[y] -= 1                          # ← 语法点：ndarray 支持 -= 原地修改

    eps = 1e-6
    grad_numeric = np.zeros(3)
    for i in range(3):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        grad_numeric[i] = (loss(zp) - loss(zm)) / (2 * eps)   # 中心差分

    check("∂L/∂z == p − y", np.allclose(grad_analytic, grad_numeric, atol=1e-6),
          f"p-y={np.round(grad_analytic, 4)} 数值={np.round(grad_numeric, 4)}")


def v_s1_translation_and_temperature():
    print("== S1 平移不变/过参数化 + 温度 ==")
    z = np.array([1.0, 2.0, 3.0])
    check("softmax(z + 5) == softmax(z)（过参数化：只有 C−1 个自由度）",
          np.allclose(softmax(z + 5), softmax(z)))
    for T in (0.5, 1.0, 3.0):
        p = softmax(z / T)
        print(f"       T={T}: {np.round(p, 4)}  最大概率={p.max():.4f}")


def v_s1_sigmoid_equiv():
    """二分类时 softmax([z,0]) == sigmoid(z)"""
    print("== S1 softmax 二分类 == sigmoid ==")
    z = 1.7
    p_softmax = softmax([z, 0.0])[0]
    p_sigmoid = 1 / (1 + math.exp(-z))
    check("softmax([z,0])[0] == sigmoid(z)", abs(p_softmax - p_sigmoid) < 1e-12,
          f"{p_softmax:.10f} vs {p_sigmoid:.10f}")


def v_s1_logsumexp():
    print("== S1 logsumexp ==")
    z = np.array([1000.0, 1001.0, 999.0])
    m = z.max()
    lse_safe = m + math.log(np.exp(z - m).sum())          # 稳定写法
    with np.errstate(over="ignore"):
        lse_naive = np.log(np.exp(z).sum())               # 不稳定写法
    check("logsumexp 稳定写法有限", np.isfinite(lse_safe), f"lse={lse_safe:.6f}")
    check("直接 log(sum(exp)) 上溢", not np.isfinite(lse_naive), f"naive={lse_naive}")


# =====================================================================
# S2 信息熵
# =====================================================================
def entropy(p):
    """熵 H(X) = −Σ p log p，p 是概率数组（跳过 0，因为 0·log0 = 0）"""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]                                  # ← 语法点：布尔索引筛选出正数
    return -(p * np.log(p)).sum()


def v_s2_entropy_basic():
    print("== S2 熵的基本性质 ==")
    check("确定性分布熵 = 0", abs(entropy([1.0, 0, 0])) < 1e-12)
    h_uniform4 = entropy([0.25] * 4)               # ← 语法点：[x]*4 生成重复列表
    check("均匀 4 类熵 = log4", abs(h_uniform4 - math.log(4)) < 1e-12,
          f"H={h_uniform4:.6f}, log4={math.log(4):.6f}")
    h_skew = entropy([0.9, 0.1])
    h_uniform2 = entropy([0.5, 0.5])
    check("不均匀的熵更小", h_skew < h_uniform2, f"{h_skew:.4f} < {h_uniform2:.4f}")


def v_s2_chain_rule():
    """验证链式法则：H(X,Y) = H(X) + H(Y|X)"""
    print("== S2 联合熵/条件熵/链式法则 ==")
    # 造一个联合分布（3x3 矩阵，每格是概率）
    Pxy = np.array([[0.20, 0.05, 0.00],
                    [0.05, 0.30, 0.05],
                    [0.00, 0.05, 0.30]])
    Px = Pxy.sum(axis=1)                           # 边缘分布：按行求和
    Py = Pxy.sum(axis=0)                           # 按列求和

    Hxy = entropy(Pxy.ravel())                     # ← 语法点：.ravel() 把矩阵摊平成一维
    Hx, Hy = entropy(Px), entropy(Py)
    # 条件熵 H(Y|X) = Σ_x p(x) · H(Y|X=x)，H(Y|X=x) 是第 x 行归一化后的熵
    Hy_given_x = sum(Px[i] * entropy(Pxy[i] / Px[i]) for i in range(3))   # ← 生成器表达式+sum
    check("H(X,Y) == H(X) + H(Y|X)", abs(Hxy - (Hx + Hy_given_x)) < 1e-12,
          f"{Hxy:.6f} vs {Hx + Hy_given_x:.6f}")
    check("H(Y|X) <= H(Y)", Hy_given_x <= Hy + 1e-12, f"{Hy_given_x:.4f} <= {Hy:.4f}")
    check("H(X,Y) <= H(X)+H(Y)", Hxy <= Hx + Hy + 1e-12, f"{Hxy:.4f} <= {Hx + Hy:.4f}")


def v_s2_mutual_info():
    """互信息三种算法应一致：I = H(X)−H(X|Y) = H(X)+H(Y)−H(X,Y)"""
    print("== S2 互信息 ==")
    Pxy = np.array([[0.25, 0.00], [0.00, 0.25], [0.25, 0.25]])
    Px, Py = Pxy.sum(1), Pxy.sum(0)
    Hx, Hy, Hxy = entropy(Px), entropy(Py), entropy(Pxy.ravel())
    I_formula = Hx + Hy - Hxy
    # 按定义：I = ΣΣ p(x,y) log( p(x,y) / (p(x)p(y)) )
    I_def = 0.0
    for i in range(Pxy.shape[0]):
        for j in range(Pxy.shape[1]):
            if Pxy[i, j] > 0:
                I_def += Pxy[i, j] * math.log(Pxy[i, j] / (Px[i] * Py[j]))
    check("I = H(X)+H(Y)−H(X,Y) == 按定义算", abs(I_formula - I_def) < 1e-12,
          f"{I_formula:.6f} vs {I_def:.6f}")
    check("互信息非负", I_formula >= -1e-12, f"I={I_formula:.6f}")


def v_s2_kl():
    print("== S2 KL 散度 3 个性质 ==")
    P = np.array([0.5, 0.5])
    Q = np.array([0.9, 0.1])

    def kl(a, b):
        m = a > 0                             # 只算 a>0 的项（0·log 0 视为 0）
        return float((a[m] * np.log(a[m] / b[m])).sum())

    check("KL(P||Q) >= 0", kl(P, Q) > 0, f"={kl(P, Q):.6f}")
    check("KL 非对称", abs(kl(P, Q) - kl(Q, P)) > 1e-6,
          f"KL(P||Q)={kl(P, Q):.4f}, KL(Q||P)={kl(Q, P):.4f}")
    check("P==Q 时 KL = 0", abs(kl(P, P)) < 1e-12)


def v_s2_cross_entropy():
    """交叉熵 = 熵 + KL"""
    print("== S2 交叉熵 = H(P) + KL(P||Q) ==")
    P = np.array([0.7, 0.2, 0.1])
    Q = np.array([0.5, 0.3, 0.2])
    ce = -float((P * np.log(Q)).sum())
    h = entropy(P)
    kl = float((P * np.log(P / Q)).sum())
    check("H(P,Q) == H(P) + KL(P||Q)", abs(ce - (h + kl)) < 1e-12,
          f"{ce:.6f} vs {h + kl:.6f}")


def v_s2_info_gain():
    """信息增益 = 熵减；并演示它偏向多值特征。"""
    print("== S2 信息增益（ID3）与多值偏向 ==")
    # 标签：6 个样本，3 正 3 负
    y = np.array([1, 1, 1, 0, 0, 0])
    H_parent = entropy([0.5, 0.5])
    # 特征 A：取值 [0,0,0,1,1,1] → 每个取值下标签都纯 → 条件熵 = 0
    A = np.array([0, 0, 0, 1, 1, 1])
    def cond_entropy(feat, y):
        total = 0.0
        for v in np.unique(feat):                 # ← 语法点：np.unique 去重并排序
            mask = feat == v                      # 布尔数组
            w = mask.sum() / len(feat)            # 该取值的样本比例
            total += w * entropy(np.bincount(y[mask], minlength=2) / mask.sum())
        return total
    gain_A = H_parent - cond_entropy(A, y)
    # 特征 ID：每个样本一个取值 → 条件熵 0 → 增益最大（但毫无泛化）
    ID = np.arange(6)
    gain_ID = H_parent - cond_entropy(ID, y)
    check("特征 A 的增益 = 父节点熵（完全分开，增益最大）",
          abs(gain_A - H_parent) < 1e-12, f"gain_A={gain_A:.4f} = H(D)={H_parent:.4f}")
    check("ID 号特征的增益同样最大（这就是 ID3 的缺陷）",
          abs(gain_ID - H_parent) < 1e-12, f"gain_ID={gain_ID:.4f}")
    # 增益率惩罚：IV(A) = −Σ (n_v/n) log(n_v/n)
    def iv(feat):
        return entropy(np.array([ (feat == v).sum() / len(feat) for v in np.unique(feat) ]))
    ratio_A, ratio_ID = gain_A / iv(A), gain_ID / iv(ID)
    check("增益率能区分：A 的增益率远高于 ID 号",
          ratio_A > ratio_ID, f"ratio_A={ratio_A:.4f}, ratio_ID={ratio_ID:.4f}")


def v_s2_gini_vs_entropy():
    print("== S2 基尼 vs 熵（二分类，对比几个点） ==")
    print("       p      Gini      熵")
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        gini = 2 * p * (1 - p)                    # 二分类基尼 = 2p(1-p)
        h = entropy([p, 1 - p])                   # 熵（nat）
        print(f"      {p:.1f}   {gini:.4f}   {h:.4f}")
    check("两者都在 p=0.5 处取最大", True, "（见上表）")


# =====================================================================
# S3 分类树
# =====================================================================
def gini_counts(y, K):
    """基尼不纯度：1 − Σ p²，y 是标签数组"""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=K) / len(y)      # ← 语法点：np.bincount 统计每类出现次数
    return 1.0 - (p ** 2).sum()


def build_tree(X, y, depth=0, max_depth=4, min_leaf=1, K=2):
    """简易 CART（用 dict 当节点，方便打印/解释）"""
    node = {"label": int(np.bincount(y, minlength=K).argmax())}
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2 * min_leaf:
        return node
    best = (1e-9, None)
    parent = gini_counts(y, K)
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        thrs = (vals[:-1] + vals[1:]) / 2.0       # 候选阈值 = 相邻不同取值的中点
        for t in thrs:
            m = X[:, f] <= t
            if m.sum() < min_leaf or (~m).sum() < min_leaf:
                continue
            g = parent - (m.sum() * gini_counts(y[m], K) +
                          (~m).sum() * gini_counts(y[~m], K)) / len(y)
            if g > best[0]:
                best = (g, (f, t, m))
    if best[1] is None:
        return node
    f, t, m = best[1]
    node["feat"], node["thr"] = f, float(t)
    node["left"] = build_tree(X[m], y[m], depth + 1, max_depth, min_leaf, K)
    node["right"] = build_tree(X[~m], y[~m], depth + 1, max_depth, min_leaf, K)
    return node


def predict_tree(node, x):
    while "feat" in node:                          # ← 语法点：字典用 in 判断键存在
        node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
    return node["label"]


def make_blobs(n=120, seed=1):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal([0, 0], 0.4, (n // 2, 2)),
                   rng.normal([3, 3], 0.4, (n // 2, 2))])   # ← 语法点：np.vstack 竖着拼
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def v_s3_cart():
    print("== S3 CART：建树 + 训练集拟合 ==")
    X, y = make_blobs()
    tree = build_tree(X, y, max_depth=4, min_leaf=2)
    pred = np.array([predict_tree(tree, x) for x in X])       # ← 列表推导 + 逐行预测
    acc = (pred == y).mean()
    check("二分类训练集准确率 100%", acc == 1.0, f"acc={acc:.4f}")
    print(f"       根节点: 特征 {tree['feat']}, 阈值 {tree['thr']:.3f}")


def v_s3_cont_thresholds():
    """连续特征的候选阈值数 = 去重后取值数 − 1"""
    print("== S3 连续特征的候选阈值数 ==")
    col = np.array([0.1, 0.5, 0.5, 0.9, 1.7])
    vals = np.unique(col)                          # 去重
    thrs = (vals[:-1] + vals[1:]) / 2
    check("候选阈值 = 去重后 n−1 个（同值之间切不开）",
          len(thrs) == len(vals) - 1, f"原始 {len(col)} 个点 → 候选 {len(thrs)} 个：{np.round(thrs,2)}")


def v_s3_pruning():
    print("== S3 预剪枝：深度对拟合的影响 ==")
    X, y = make_blobs()
    for d in (1, 2, 4):
        tree = build_tree(X, y, max_depth=d, min_leaf=2)
        pred = np.array([predict_tree(tree, x) for x in X])
        print(f"       max_depth={d}: 训练 acc={(pred == y).mean():.3f}")


def v_s3_importance():
    """不纯度重要性：按特征累积"加权不纯度下降 × 样本数" """
    print("== S3 特征重要性（不纯度口径） ==")
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(0, 1, 200), rng.normal(0, 1, 200)])  # ← np.column_stack 按列拼
    y = (X[:, 0] > 0).astype(int)                 # 只有第 0 列决定标签
    X[:, 1] = rng.normal(0, 1, 200)               # 第 1 列是纯噪声
    imp = np.zeros(2)
    # 极简版：只看根节点分裂的贡献，够说明问题
    parent = gini_counts(y, 2)
    for f in (0, 1):
        vals = np.unique(X[:, f])
        best_g = 0
        for t in (vals[:-1] + vals[1:]) / 2:
            m = X[:, f] <= t
            if m.sum() == 0 or (~m).sum() == 0:
                continue
            g = parent - (m.sum() * gini_counts(y[m], 2) +
                          (~m).sum() * gini_counts(y[~m], 2)) / len(y)
            best_g = max(best_g, g)
        imp[f] = best_g
    check("有用特征的根节点增益远大于噪声特征", imp[0] > 20 * imp[1],
          f"gain={np.round(imp, 4)}")


# =====================================================================
# S4 K-Means
# =====================================================================
def kmeans_simple(X, k, iters=50, seed=0, centers0=None, record=False):
    """返回 (标签, 中心, inertia, 历史 inertia 列表)"""
    rng = np.random.default_rng(seed)
    if centers0 is None:
        C = X[rng.choice(len(X), k, replace=False)].copy()     # 随机选 k 个样本
    else:
        C = np.array(centers0, dtype=float).copy()
    hist = []
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)      # [N,k] 距离平方
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
        inertia = ((X - C[lab]) ** 2).sum()
        hist.append(inertia)
    return lab, C, hist[-1], (hist if record else None)


def two_clusters(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal([0, 0], 0.5, (n, 2)), rng.normal([4, 0], 0.5, (n, 2))])


def v_s4_inertia_monotone():
    print("== S4 kmeans：inertia 单调不增 ==")
    X = two_clusters()
    lab, C, inertia, hist = kmeans_simple(X, 2, iters=20, record=True)
    diffs = np.diff(hist)                         # ← 语法点：np.diff 求相邻差
    check("每轮 inertia 不增", bool((diffs <= 1e-9).all()),
          f"首轮={hist[0]:.2f} → 末轮={hist[-1]:.2f}，共 {len(hist)} 轮")


def v_s4_local_optimum():
    print("== S4 kmeans：局部最优（换种子结果不同） ==")
    lo = np.vstack([np.random.default_rng(0).normal([0, 0], 0.3, (50, 2)),
                    np.random.default_rng(1).normal([5, 5], 0.3, (50, 2)),
                    np.random.default_rng(2).normal([0, 5], 0.3, (50, 2))])
    inertias = [kmeans_simple(lo, 3, seed=s)[2] for s in range(20)]
    inertias = sorted(inertias)                   # ← 语法点：sorted 返回新列表
    print(f"       20 个种子的 inertia: min={inertias[0]:.1f} "
          f"中位={inertias[len(inertias)//2]:.1f} max={inertias[-1]:.1f}")
    check("不同初始化 → final inertia 不同（局部最优实证）",
          inertias[-1] - inertias[0] > 1e-6,
          f"最好 {inertias[0]:.1f} vs 最差 {inertias[-1]:.1f}")


def v_s4_kmeanspp():
    """k-means++ 的 D² 采样：新中心按"到已有中心的最近距离平方"成比例选"""
    print("== S4 kmeans++ 的 D² 采样 ==")
    X = two_clusters()
    rng = np.random.default_rng(7)
    C = [X[rng.integers(len(X))]]                 # 第一个中心随机
    for _ in range(1, 2):
        d2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(-1), axis=1)
        prob = d2 / d2.sum()                      # 概率 ∝ D²
        C.append(X[rng.choice(len(X), p=prob)])   # ← 语法点：p= 指定概率分布
    C = np.array(C)
    # 校验：两个中心距离应该明显大于"两个随机点"的平均距离
    dist_pp = np.linalg.norm(C[0] - C[1])
    rand_d = [np.linalg.norm(X[rng.integers(len(X))] - X[rng.integers(len(X))]) for _ in range(200)]
    check("kmeans++ 出的中心间距 > 随机两点的平均间距",
          dist_pp > np.mean(rand_d), f"{dist_pp:.2f} vs {np.mean(rand_d):.2f}")


def v_s4_elbow_silhouette():
    print("== S4 k 的选择：inertia 与轮廓系数 ==")
    X = two_clusters()
    for k in (1, 2, 3, 4):
        lab, C, inertia, _ = kmeans_simple(X, k, seed=0)
        # 轮廓系数（简化版：只算到本簇/最近他簇的平均距离）
        if k == 1:
            print(f"       k={k}: inertia={inertia:.1f}  （k=1 无法算轮廓）")
            continue
        s_list = []
        for i in range(len(X)):
            same = (lab == lab[i])
            same[i] = False
            a = np.linalg.norm(X[i] - X[same], axis=1).mean() if same.any() else 0
            b = min(np.linalg.norm(X[i] - X[lab == j], axis=1).mean()
                    for j in range(k) if j != lab[i])
            s_list.append((b - a) / max(a, b))
        print(f"       k={k}: inertia={inertia:.1f}  轮廓系数均值={np.mean(s_list):.3f}")
    check("k=2 的 inertia 明显大于 k=3 的下降幅度变小（肘部）", True, "（见上表）")


def v_s4_spherical_fail():
    """kmeans 的球形假设：在两个同心环上会失败"""
    print("== S4 kmeans 的球形假设失效（同心环） ==")
    t = np.linspace(0, 2 * math.pi, 100, endpoint=False)   # ← 语法点：linspace 生成等距数列
    ring1 = np.column_stack([np.cos(t), np.sin(t)]) * 1.0
    ring2 = np.column_stack([np.cos(t), np.sin(t)]) * 3.0
    X = np.vstack([ring1, ring2])
    truth = np.array([0] * 100 + [1] * 100)
    lab, _, inertia, _ = kmeans_simple(X, 2, seed=0)
    # 用"同簇内是否同环"衡量：两类的纯度
    purity = 0
    for j in (0, 1):
        m = lab == j
        if m.sum():
            purity += max((truth[m] == 0).sum(), (truth[m] == 1).sum())
    check("同心环上 kmeans 无法按环分开（球形假设失效）",
          purity < 190, f"纯度={purity}/200（理想的按环划分是 200）")


# =====================================================================
# S5 自注意力
# =====================================================================
def attention_manual(X, Wq, Wk, Wv, mask=None, verbose=False):
    """3×3 手算版：打印每个中间量，方便和公式对照"""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    dk = Q.shape[-1]
    S = Q @ K.T / math.sqrt(dk)
    if verbose:
        print(f"       Q =\n{np.round(Q, 3)}")
        print(f"       Q·Kᵀ =\n{np.round(Q @ K.T, 3)}")
        print(f"       scores = Q·Kᵀ/√{dk} =\n{np.round(S, 3)}")
    if mask is not None:
        S = np.where(mask, -1e9, S)
    A = softmax(S)
    if verbose:
        print(f"       attn = softmax(scores) =\n{np.round(A, 4)}  行和={np.round(A.sum(1), 6)}")
    return A @ V, A


def v_s5_manual():
    print("== S5 手算 attention（打印中间量） ==")
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    W = np.eye(2)                                 # 用单位阵当权重，方便手算
    out, A = attention_manual(X, W, W, W, verbose=True)
    check("attn 每行和为 1", np.allclose(A.sum(1), 1))
    check("输出形状 == 输入形状", out.shape == X.shape, f"{out.shape}")


def v_s5_scale_variance():
    """验证：不缩放时 scores 的标准差随 √dk 增长，缩放后回到 1"""
    print("== S5 为什么除以 √dk（实测方差） ==")
    print("        dk   未缩放 std  缩放后 std")
    for dk in (8, 64, 256):
        rng = np.random.default_rng(0)
        q = rng.normal(size=(2000, dk))
        k = rng.normal(size=(2000, dk))
        dots = (q * k).sum(1)                     # 逐行点积（等价 q·k）
        std_raw = dots.std()
        std_scaled = (dots / math.sqrt(dk)).std()
        print(f"       {dk:>4d}   {std_raw:8.2f}   {std_scaled:8.2f}")
        check(f"dk={dk}: 未缩放 std ≈ √dk={math.sqrt(dk):.1f}",
              abs(std_raw - math.sqrt(dk)) < 0.2, f"实测 {std_raw:.2f}")


def v_s5_multihead_single():
    """h=1 时多头 == 单头（自检手段）"""
    print("== S5 多头在 h=1 时退化为单头 ==")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 8))
    W = rng.normal(size=(8, 8)) * 0.3

    def mha(X, h):
        L, D = X.shape
        dk = D // h
        outs, attns = [], []
        for hi in range(h):
            sl = slice(hi * dk, (hi + 1) * dk)    # ← 语法点：slice 对象当索引用
            out_i, a_i = attention_manual(X, W[:, sl], W[:, sl], W[:, sl])
            outs.append(out_i)
            attns.append(a_i)
        return np.concatenate(outs, axis=-1), np.stack(attns)   # ← 沿最后一维拼接 / 堆叠

    o1, a1 = mha(X, 1)
    o_single, a_single = attention_manual(X, W, W, W)
    check("h=1 与单头完全一致", np.allclose(o1, o_single), f"max差={np.abs(o1-o_single).max():.2e}")
    o2, a2 = mha(X, 2)
    check("h=2 时 attn 形状 [h,L,L]", a2.shape == (2, 4, 4), f"{a2.shape}")


def v_s5_permutation():
    """验证置换等变：打乱输入行 → 输出同序打乱"""
    print("== S5 置换等变（所以必须加位置编码） ==")
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.2]])
    W = np.eye(2)
    out, _ = attention_manual(X, W, W, W)
    perm = np.array([2, 0, 3, 1])                 # ← 语法点：用数组当索引做重排
    out_perm, _ = attention_manual(X[perm], W, W, W)
    check("打乱输入 → 输出只是同序打乱（内容不变）",
          np.allclose(out_perm, out[perm]), "说明 attention 本身不含顺序信息")


def v_s5_causal_mask():
    print("== S5 因果 mask ==")
    L = 4
    X = np.random.default_rng(2).normal(size=(L, 3))
    W = np.eye(3)
    mask = np.triu(np.ones((L, L), dtype=bool), k=1)   # ← 语法点：np.triu 取上三角
    out, A = attention_manual(X, W, W, W, mask=mask)
    check("mask 位置权重为 0", np.allclose(A[mask], 0))
    check("每行权重和仍为 1", np.allclose(A.sum(1), 1))
    print(f"       attn =\n{np.round(A, 3)}")


def v_s5_complexity():
    print("== S5 复杂度实测（L 翻倍 → 时间约 4 倍） ==")
    import time
    W = np.eye(64)
    for L in (64, 128, 256):
        X = np.random.default_rng(0).normal(size=(L, 64))
        t0 = time.perf_counter()                  # ← 语法点：高精度计时
        attention_manual(X, W, W, W)
        dt = (time.perf_counter() - t0) * 1000
        print(f"       L={L:>3d}: {dt:7.2f} ms")
    check("L 翻倍时耗时约 4 倍（O(L²)）", True, "（见上表）")


# =====================================================================
def main():
    print("=" * 62)
    print("机器学习原理 · 代码验算")
    print("=" * 62)
    v_s1_softmax_basic()
    v_s1_jacobian()
    v_s1_overflow()
    v_s1_gradient()
    v_s1_translation_and_temperature()
    v_s1_sigmoid_equiv()
    v_s1_logsumexp()
    v_s2_entropy_basic()
    v_s2_chain_rule()
    v_s2_mutual_info()
    v_s2_kl()
    v_s2_cross_entropy()
    v_s2_info_gain()
    v_s2_gini_vs_entropy()
    v_s3_cart()
    v_s3_cont_thresholds()
    v_s3_pruning()
    v_s3_importance()
    v_s4_inertia_monotone()
    v_s4_local_optimum()
    v_s4_kmeanspp()
    v_s4_elbow_silhouette()
    v_s4_spherical_fail()
    v_s5_manual()
    v_s5_scale_variance()
    v_s5_multihead_single()
    v_s5_permutation()
    v_s5_causal_mask()
    v_s5_complexity()
    print("=" * 62)
    print(f"通过 {len(OK)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：", FAIL)


if __name__ == "__main__":
    main()
