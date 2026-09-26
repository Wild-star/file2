# FusionWarp-Stereo: Correlation Anchors and Gated Fusion for ViT-based Iterative Stereo Matching

**Abstract**

Iterative stereo matching has split into two paradigms of correspondence representation: explicit matching search via correlation volumes, and local residual refinement via warped features. WAFT-Stereo recently showed that a *pure warping* paradigm—without any cost volume—can achieve state-of-the-art accuracy by relying on a frozen monocular visual foundation model (VFM) and a ViT-based iterative decoder. However, WAFT discards explicit matching evidence entirely, leaving the ViT decoder to implicitly re-derive correspondence, which is costly and slow to converge in textureless or large-disparity regions. Meanwhile, WAVE-Stereo unified correlation and warping, but in a *lightweight, ConvGRU-based, VFM-free* regime. This paper studies the orthogonal quadrant that WAVE-Stereo leaves open: **how to inject explicit matching evidence into a VFM + ViT-based warping pipeline**. We propose FusionWarp-Stereo, which augments WAFT with three learnable modules: (1) a lightweight **MatchingBranch** that extracts matching-friendly features directly from raw images, (2) a **GlobalMatcher** with learnable QKV cross-attention and full-range correlation that regresses an initial disparity and a global-context feature, and (3) a **SparseCorrAnchor** that injects *non-aggregated* group-wise correlation into the iterative update, fused with the global context through a learnable **GatedFusion**. We additionally equip the ViT decoder with a **TokenSparseVitIter** that learns token-level saliency gating. All modules are zero-initialized so that loading a pretrained WAFT checkpoint yields bit-identical behavior. Preliminary synthetic ablations suggest that *non-aggregated* correlation anchors outperform aggregated 3D cost volumes, and the added modules cost only +2.0% FLOPs and +0.2% memory at +0.11% parameters. Full-scale training is ongoing.

---

## 1. Introduction

Stereo matching estimates dense disparity from a rectified pair, serving as a core task in autonomous driving, robotics, and augmented reality. The field has been dominated by cost-volume methods [PSMNet, GwcNet, ACVNet, IGEV], which construct explicit 4D/3D correlation tensors and regularize them with 3D convolutions. This is accurate but computationally and memory expensive.

RAFT-Stereo introduced an iterative-optimization paradigm using a ConvGRU that sequentially refines disparity from a precomputed correlation pyramid. IGEV-Stereo added a geometry-encoding volume. These methods keep *explicit matching evidence* in the form of correlation features.

A recent line questions whether cost volumes are necessary at all. **WAFT-Stereo** demonstrated that a *pure warping* algorithm—warping the right feature with the current disparity and iteratively correcting residuals with a ViT decoder on top of a frozen monocular VFM (Depth-Anything-V2 / DINOv3)—can reach top accuracy on ETH3D, KITTI, and Middlebury while being 1.8–6.7× faster than competitive methods. However, WAFT discards explicit matching evidence: its encoder features are *not* matching costs (we verify this in §4), and all correspondence reasoning is implicitly packed into the ViT decoder. This makes convergence slower and less robust in textureless, repetitive, and large-disparity regions.

**WAVE-Stereo** observed that correlation volumes and feature warping are *complementary* representations and unified them in a GeoWarp Correspondence Encoder (GWCE), together with Periodic Global Context Propagation (PGCP) to compensate the local receptive field of the ConvGRU. Critically, WAVE-Stereo operates in a **lightweight, VFM-free, ConvGRU-based** regime: a MobileNetV2 backbone at 1/4 resolution, achieving real-time 66 ms but lower accuracy (ETH3D Bad-2.0 0.86 vs. WAFT's 0.32).

The quadrant that WAVE-Stereo leaves open is the one where WAFT already excels: **the VFM + ViT-based, high-accuracy regime**. In this paper we ask: *if we inject explicit matching evidence into a VFM + ViT warping pipeline, does it help, and in what form should that evidence be?* We make the following contributions:

1. **FusionWarp-Stereo**, a ViT-based iterative stereo framework that augments WAFT with three learnable modules—MatchingBranch, GlobalMatcher, and SparseCorrAnchor—plus GatedFusion and a token-sparse ViT decoder, all zero-initialized for surgical safety (loading a pretrained WAFT is bit-identical).

2. A **positioning as a cross-regime study** with three falsifiable hypotheses (Table 1): (H1) *non-aggregated* correlation outperforms aggregated 3D cost volumes; (H2) the ViT's native global attention makes periodic global-context propagation (PGCP) redundant; (H3) a small matching branch suffices to make VFM features matching-friendly.

3. **Preliminary synthetic ablations** (§4.2) supporting H1, and **efficiency analysis** (§4.3) showing the added modules cost only +2.0% FLOPs and +0.2% memory at +0.11% parameters.

| Hypothesis | WAVE-Stereo | FusionWarp (ours) |
|---|---|---|
| Matching evidence | geometry-encoding volume (aggregated) | non-aggregated correlation anchor |
| Global context | PGCP (periodic, patch) | ViT native (per-iteration) |
| Backbone | MobileNetV2 (VFM-free) | DAv2/DINOv3 (VFM) |

**Table 1.** The three falsifiable hypotheses that distinguish this work from WAVE-Stereo.

---

## 2. Related Work

**Cost-volume stereo.** PSMNet [1] built 4D cost volumes regularized by 3D hourglass networks. GwcNet [2] introduced group-wise correlation. ACVNet [3] used attention concatenation. IGEV-Stereo [4] constructed a geometry-encoding volume combining matching and geometry. These methods dominate in-domain accuracy but are heavy.

**Iterative refinement.** RAFT-Stereo [5] adopted a ConvGRU over a correlation pyramid. CREStereo [6] improved generalization. Selective-Stereo [7] and Selective-IGEV update only selected regions. SEA-RAFT [8] used a mixture-of-Laplace loss for robustness.

**Global matching.** GMFlow [9] used all-pairs softmax matching and demonstrated that a 4D volume is not necessary if matching is framed as a search problem. STTR [10] used cross-attention for correspondence.

**Warping-only and foundation models.** WAFT-Stereo [11] showed that iterative warping with a frozen VFM (Depth-Anything-V2 [12], DINOv3 [13]) and a LoRA-adapted ViT decoder eliminates cost volumes entirely. MonSter and FoundationStereo [14] also exploit monocular VFM priors.

**Unifying correlation and warping.** WAVE-Stereo [15] proposed GWCE to unify correlation search, residual alignment, and disparity prior at the ConvGRU input, plus PGCP for periodic global context. Our work differs in the *regime*: we target the VFM + ViT quadrant WAVE-Stereo explicitly avoids, and we inject *non-aggregated* correlation, arguing (H1) that aggregation is unnecessary—and possibly harmful—when the decoder already has global attention.

---

## 3. Method

### 3.1 Preliminaries: WAFT-Stereo

Given rectified left/right images $I_L, I_R$, a shared encoder (DAv2 or DINOv3, frozen with LoRA) produces context features $f_1, f_2$ and a hidden state $net$ at 1/2 resolution. An initial disparity is produced by a *prop* branch that classifies disparity into bins with soft-argmax. Then $T$ iterations refine it:

$$
\begin{aligned}
d &\leftarrow \text{detach}(d), \\
\tilde{f}_2 &= \text{warp}(f_2, d), \\
net &= \delta\text{-proj}\big([\,f_1,\ \tilde{f}_2,\ net,\ d\,]\big), \\
net &= \delta\text{-decoder}(net), \\
d &\leftarrow d + \text{disp-head}(net),
\end{aligned}
$$

followed by RAFT-style convex upsampling. The key observation is that the update input $[f_1, \tilde{f}_2, net, d]$ contains **no explicit matching evidence**—the only cross-view signal is the warped feature $\tilde{f}_2$. The ViT decoder must implicitly recover correspondence.

### 3.2 MatchingBranch

We found (verify in §4) that VFM encoder features are *not* matching costs: after aligning with ground-truth disparity, the argmax of cosine similarity falls on the correct location only 46–55% of the time. We therefore add a lightweight CNN `MatchingBranch` (32 channels, stride 4 → 1/4 resolution) that extracts *matching-friendly* features $m_1, m_2$ directly from the raw images. It has only ~30K parameters.

### 3.3 GlobalMatcher

On 1/8-scale features, a **learnable QKV cross-attention** layer (left↔right, shared weights) enhances features against textureless/occlusion ambiguity, followed by **full-range correlation** with soft-argmax:

$$
C(x,d) = \langle \hat{f}_1(x), \hat{f}_2(x-d)\rangle, \qquad d_{gm} = \sum_d d \cdot \text{softmax}_d(C(x,d)).
$$

$d_{gm}$ is upsampled to 1/2 resolution as the initial disparity (replacing the prop bins), and the cross-attended features are projected into a global-context feature $g_{feat}$ (1/2 resolution) for the fusion module below.

### 3.4 SparseCorrAnchor

At each iteration, within a narrow band $\pm R$ around the current disparity, we compute **group-wise correlation** (channels split into $G$ groups, normalized dot product) between $m_1$ and warped $m_2$:

$$
A(x) = \Big[\text{gcorr}_G\big(m_1(x), \text{warp}(m_2, d(x)+o)\big)\Big]_{o=-R}^{R},
$$

aggregated by a zero-initialized $3\times3$ convolution into an anchor feature added to the $\delta$-proj output. Unlike WAVE-Stereo's geometry-encoding volume, we apply **no 3D/2D aggregation** over the disparity dimension (H1). A `GEVCostAnchor` variant with separable 3D aggregation is kept for controlled ablation.

### 3.5 GatedFusion

The anchor (local matching evidence) and the global context $g_{feat}$ are fused by a learnable spatial gate:

$$
g = \sigma\big(\text{conv}([\,A,\ g_{feat}\,])\big), \qquad F = g \odot A + (1-g) \odot g_{feat},
$$

whose output is scaled by a zero-initialized parameter so that $F=0$ at initialization (surgical safety).

### 3.6 TokenSparseVitIter

The WAFT $\delta$-decoder is a timm ViT with LoRA. We augment it with a token-level saliency gate after each block:

$$
h' = tok + \gamma \odot \big(\text{blk}(tok) - tok\big), \qquad \gamma = 2\,\sigma(\text{saliency}(tok)),
$$

where `saliency` is a zero-initialized linear layer, so $\gamma=1$ at initialization (bit-identical to the plain ViT). This gives the ViT a selective-update capability that a ConvGRU (as in WAVE-Stereo) cannot express (H2).

### 3.7 Training Objective

We keep WAFT's mixture-of-Laplace loss with exponential weighting over iterations, plus the initial-disparity KL loss. All new modules are trainable; the VFM backbone stays frozen with LoRA.

---

## 4. Experiments

### 4.1 Setup

**Datasets.** Training uses the 12 synthetic/real datasets from WAFT's config (SceneFlow, TartanAir, CREStereo, FSD, etc.); evaluation on ETH3D, KITTI 2012/2015, Middlebury. Data layout follows `docs/DATASETS_FORMAT.md`.

**Backbone.** DAv2-Small / DINOv3-Small, frozen with LoRA (rank 8, alpha 16), patch size 8, 3 delta iterations, max disparity 800.

**Metrics.** EPE, Bad-1px/3px, D1; plus FLOPs and peak memory.

### 4.2 Preliminary Ablations (synthetic POC)

> **Honest scope note:** the following numbers come from a *self-contained synthetic POC* (96×128 random-texture pairs, 120 training steps, fair initialization). They validate *mechanisms*, not final accuracy. Full-scale results are reported separately (§4.4) when training completes.

**(a) Are VFM features matching costs?** Aligning DAv2 features with GT disparity, the argmax of cosine similarity lands on the correct offset only 46–55% of the time, versus 87% after training a 34K-parameter MatchingBranch. This confirms that matching evidence must be *learned*, not read off the frozen VFM.

**(b) Does matching evidence help?** Table 2 ablates the anchor on the synthetic POC. Any anchor improves over the pure-warp baseline, and the *non-aggregated* correlation anchor is best—supporting H1.

| Config | Params | EPE (last) | rel. to no_anchor |
|---|---|---|---|
| no_anchor (pure warp) | 269.6K | 2.430 | — |
| corr (non-aggregated) | 269.6K | **1.850** | **+23.9%** |
| gev-full3d (aggregated) | 254.0K | 2.092 | +13.9% |
| gev-sep3d (separable 3D) | 252.5K | 1.937 | +20.3% |

**Table 2.** Synthetic POC anchor ablation. Non-aggregated correlation beats 3D aggregation (H1).

**(c) GlobalMatcher initial disparity.** On the synthetic POC, replacing bins classification with GlobalMatcher's full-range regression lowers initial EPE from ~28 px to ~23 px at random init, indicating a better initialization for convergence.

### 4.3 Efficiency Analysis (measured)

Measured at 480×640, batch 1, RTX 5060:

| Config | FLOPs | Peak memory | Params |
|---|---|---|---|
| WAFT baseline | 826.0 G | 1948 MB | 77.05M |
| +corr anchor | 832.4 G (+0.78%) | 1952 MB (+0.19%) | 77.11M |
| full fusion | 842.7 G (+2.02%) | 1952 MB (+0.19%) | 77.12M |

The per-module breakdown: MatchingBranch 1.43 G, SparseCorrAnchor 1.19 G, GlobalMatcher 9.87 G (dominated by 100 full-range warp+corr), GatedFusion 0.13 G. The GlobalMatcher is the single FLOPs hotspot (59% of the fusion overhead) and can be reduced by lowering the disparity candidates or vectorizing the correlation.

### 4.4 Full-Scale Training (ongoing)

Full-scale end-to-end training on the 12-dataset mix (400K steps, 8 GPUs) is in progress. We will report: (i) zero-shot EPE on ETH3D/KITTI/Middlebury for each module configuration; (ii) the H1/H2/H3 hypothesis tests at scale; (iii) convergence curves with/without the GlobalMatcher initialization.

---

## 5. Conclusion

We presented FusionWarp-Stereo, a ViT-based iterative stereo framework that injects explicit matching evidence into the warping-only WAFT baseline through three learnable modules—MatchingBranch, GlobalMatcher, and SparseCorrAnchor—fused via a learnable gate and decoded by a token-sparse ViT. We positioned the work as a cross-regime study that complements WAVE-Stereo, with three falsifiable hypotheses. Preliminary synthetic ablations support the central claim that *non-aggregated correlation is sufficient and aggregation is unnecessary* in a ViT-based decoder, at negligible cost (+2.0% FLOPs, +0.2% memory, +0.11% parameters). Full-scale training will validate these hypotheses on real benchmarks.

---

## References

[1] Chang & Chen, "Pyramid Stereo Matching Network," CVPR 2018.
[2] Guo et al., "Group-wise Correlation Stereo Network," CVPR 2019.
[3] Xu et al., "ACVNet: Attention Concatenation Volume," CVPR 2022.
[4] Xu et al., "Iterative Geometry Encoding Volume for Stereo Matching," CVPR 2023.
[5] Lipson, Teed, Deng, "RAFT-Stereo," 3DV 2021.
[6] Li et al., "Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation," CVPR 2022.
[7] Wang et al., "Selective-Stereo," NeurIPS 2023.
[8] Wang et al., "SEA-RAFT," CVPR 2024.
[9] Xu et al., "GMFlow: Learning Optical Flow via Global Matching," CVPR 2022.
[10] Li et al., "Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective," ICCV 2021.
[11] Wang & Deng, "WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching," arXiv:2603.24836, 2026.
[12] Yang et al., "Depth Anything V2," NeurIPS 2024.
[13] Oquab et al., "DINOv3," arXiv 2025.
[14] Wen et al., "FoundationStereo," CVPR 2025.
[15] Liu et al., "WAVE-Stereo: Warp-Aligned Volume Encoding for Stereo Matching," arXiv:2607.13674, 2026.
