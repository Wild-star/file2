"""
Step -1 补丁：注入 state-independent 匹配锚（免训练验证）

假设：WAFT 的 delta 迭代唯一的证据是 warped_fmap2（用上一轮的 disp warp 出来），
      因此 state-dependent → 迭代不可剪枝、不可跳步。
      代价体/匹配代价是 state-independent 的绝对匹配证据。

本补丁在模型输出后，用【特征空间局部搜索】模拟这个锚：
    disp_lr = disp_pred / 2                     # 降到 fmap 分辨率
    for δ in [-R..R]:
        cost_δ = <fmap1, disp_warp(fmap2, disp_lr+δ)>     # 相关（similarity）
    d* = soft-argmax_δ(cost_δ)                  # 亚像素偏移
    disp_refined = disp_lr + d*
    disp_out = disp_pred + alpha * (2*disp_refined - disp_pred)   # 混合

环境变量：
    ANC_R      搜索半径（默认 4）
    ANC_T      softmax 温度（默认 0.05）
    ANC_ALPHA  混合系数（默认 1.0）
    ANC_ITERS  对第几轮迭代的输出做精化（默认 0 = 最后一轮；1 = 第1轮输出）
"""
import os
import torch
import torch.nn.functional as F

ANC_R = int(os.environ.get("ANC_R", "4"))
ANC_T = float(os.environ.get("ANC_T", "0.05"))
ANC_ALPHA = float(os.environ.get("ANC_ALPHA", "1.0"))
ANC_ITERS = int(os.environ.get("ANC_ITERS", "0"))     # 0=最后, n=第n轮(1-based)
ENABLED = os.environ.get("ANC_ON", "1") == "1"

STATS = []


def _patch():
    from algorithms import waft as waft_mod

    def anch_forward(self, sample, disp_init=None):
        output = {}
        image1 = self.normalize_image(sample["img1"])
        image2 = self.normalize_image(sample["img2"])
        padder = waft_mod.Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)

        fmap1, fmap2, net = self.encoder(torch.stack([image1, image2], dim=1))

        idx_bins_2x = torch.linspace(0, self.max_disp / 2, self.n_bins,
                                     device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)
        idx_bins_1x = torch.linspace(0, self.max_disp / 1, self.n_bins,
                                     device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)

        prop_hidden = self.prop_proj(torch.cat([fmap1, fmap2], dim=1))
        prop_hidden = self.prop_decoder(prop_hidden)
        prob_mask = .25 * self.prop_mask_head(prop_hidden)
        prob_bins = self.prop_bins_head(prop_hidden)
        prob_up = self.convex_upsample(prob_bins, prob_mask)
        output["init"] = padder.unpad(prob_up)
        prob_bins = F.softmax(prob_bins, dim=1)
        disp = torch.sum(prob_bins * idx_bins_2x, dim=1, keepdim=True)

        if disp_init is not None:
            disp = padder.pad(disp_init.unsqueeze(1))
            disp = F.interpolate(disp, scale_factor=0.5, mode="bilinear", align_corners=True) * 0.5

        delta_disp_preds, delta_info_preds = [], []
        for itr in range(self.iters):
            disp = disp.detach()
            warped = waft_mod.disp_warp(fmap2, disp, padding_mode="zeros")
            net = self.delta_proj(torch.cat([fmap1, warped, net, disp], dim=1))
            net = self.delta_decoder(net)
            info = self.delta_dist_head(net)
            delta_disp = self.delta_disp_head(net)
            mask = .25 * self.delta_mask_head(net)
            disp = disp + delta_disp
            disp_up = self.convex_upsample(disp * 2, mask)
            info_up = self.convex_upsample(info, mask)
            delta_disp_preds.append(disp_up)
            delta_info_preds.append(info_up)

        output["delta_disp_preds"] = [padder.unpad(p) for p in delta_disp_preds]
        output["delta_info_preds"] = [padder.unpad(p) for p in delta_info_preds]
        if self.iters > 0:
            output["disp_pred"] = output["delta_disp_preds"][-1].squeeze(1)
        else:
            output["disp_pred"] = torch.sum(F.softmax(output["init"], dim=1) * idx_bins_1x, dim=1)

        # ---------------- 匹配锚精化（state-independent） ----------------
        if ENABLED and self.iters > 0:
            tgt_idx = (self.iters - 1) if ANC_ITERS == 0 else min(ANC_ITERS - 1, self.iters - 1)
            # 该轮输出的 disp，恢复到 fmap 分辨率（×0.5）
            d_full = padder.unpad(delta_disp_preds[tgt_idx]).squeeze(1)      # (B,H,W) 全分辨率
            disp_lr = F.interpolate(d_full.unsqueeze(1), size=fmap2.shape[-2:],
                                    mode="bilinear", align_corners=True) * 0.5

            offsets = torch.arange(-ANC_R, ANC_R + 1, device=fmap1.device, dtype=fmap1.dtype)
            sims = []
            for o in offsets:
                w = waft_mod.disp_warp(fmap2, disp_lr + o, padding_mode="zeros")
                sims.append((fmap1 * w).sum(dim=1, keepdim=True))            # (B,1,h,w)
            sim = torch.cat(sims, dim=1)                                     # (B,2R+1,h,w)
            p = F.softmax(sim / ANC_T, dim=1)
            d_star = (p * offsets.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            disp_ref_lr = disp_lr + d_star
            disp_ref_full = F.interpolate(disp_ref_lr, size=d_full.shape[-2:],
                                          mode="bilinear", align_corners=True) * 2.0
            refined = d_full + ANC_ALPHA * (disp_ref_full.squeeze(1) - d_full)

            output["disp_pred_anchored"] = refined
            output["anchor_delta"] = (d_star * 2.0 * ANC_ALPHA)
            # 让官方评测器直接使用精化结果（原值另存便于对比）
            output["disp_pred_raw"] = output["disp_pred"]
            output["disp_pred"] = refined
            STATS.append(dict(mean_abs_delta=float(d_star.abs().mean()),
                              max_abs_delta=float(d_star.abs().max())))

        return output

    waft_mod.WAFT.forward = anch_forward

    import atexit

    @atexit.register
    def _report():
        if STATS:
            m = sum(s["mean_abs_delta"] for s in STATS) / len(STATS)
            mx = max(s["max_abs_delta"] for s in STATS)
            print(f"[Step-1] 锚修正统计: 平均|Δ|={m:.4f} px, 最大|Δ|={mx:.4f} px, 样本 {len(STATS)}", flush=True)

    print(f"[Step-1] ANC_ON={ENABLED} R={ANC_R} T={ANC_T} alpha={ANC_ALPHA} iters_idx={ANC_ITERS}", flush=True)


try:
    _patch()
except Exception as e:
    import traceback
    print(f"[Step-1] 注入失败: {e}")
    traceback.print_exc()
