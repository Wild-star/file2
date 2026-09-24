"""
P1 Token 稀疏迭代 —— 推理期补丁（sitecustomize 自动注入）

原理：WAFT 的 delta 模块每轮在 (H/16 × W/16) 个 token 上跑 12 层 ViT。
      P0① 证明更新是空间集中的（top-10% token 承担约一半更新量），
      因此可从第 2 轮起，只对「上一轮更新幅度大」的 token 跑 ViT。

实现：
  1) 替换 VitIter.forward 为支持 keep_idx 的版本
     - 未选中 token 的处理 = "恒等"（ViT 输出 = 输入），避免引入额外偏置
  2) 替换 WAFT.forward 的 delta 循环，逐轮计算 token 重要性并选 top-k

环境变量：
  P1_KEEP_RATIO  保留 token 比例（默认 1.0 = 关闭）
  P1_MIN_ITER    从第几轮开始稀疏（1-based，默认 2）
  P1_SKIP_MODE   identity | zero  （未选中 token 的处理，默认 identity）
"""
import os
import torch
import torch.nn.functional as F
from einops import rearrange

KEEP_RATIO = float(os.environ.get("P1_KEEP_RATIO", "1.0"))
MIN_ITER = int(os.environ.get("P1_MIN_ITER", "2"))
SKIP_MODE = os.environ.get("P1_SKIP_MODE", "identity")
STATS = []


def _patch():
    from model.iterative.vit import VitIter
    from algorithms import waft as waft_mod

    # ---------------- 1) 支持 keep_idx 的 VitIter.forward ----------------
    def sparse_forward(self, inp, keep_idx=None):
        x = self.init(inp)
        vx = self.patch_embed(x)                     # (B, C, h, w)
        B, C, h, w = vx.shape
        seq = rearrange(vx, "b c h w -> b (h w) c")
        N = seq.shape[1]
        sparse = keep_idx is not None and keep_idx.shape[1] < N

        if sparse:
            kc = keep_idx.shape[1]
            t = torch.gather(seq, 1, keep_idx[..., None].expand(-1, -1, C)).contiguous()
        else:
            t = seq
            kc = N

        # 与原版一致：收集 self.idx 指定的中间层特征（patch_size=8 → 4 层）
        vit_feats = []
        for i in range(len(self.blks)):
            t = self.blks[i](t)
            if i in self.idx:
                if sparse:
                    # 稀疏：未选中的 token 保持输入（identity），再散回完整网格
                    full = seq if SKIP_MODE != "zero" else torch.zeros_like(seq)
                    full = full.clone()
                    full.scatter_(1, keep_idx[..., None].expand(-1, -1, C), t)
                else:
                    full = t
                vit_feats.append(rearrange(full, "b (h w) c -> b c h w", h=h, w=w))

        STATS.append(dict(tokens=N, kept=kc, ratio=kc / N))

        vit_feats = self.proj(vit_feats)
        vit_feats = self.upsample(vit_feats)
        res_x = self.res_convs(x)
        return self.final_mlp(torch.cat([vit_feats[0], res_x, inp], dim=1))

    VitIter.forward = sparse_forward

    # ---------------- 2) 稀疏感知的 WAFT.forward ----------------
    orig_forward = waft_mod.WAFT.forward

    def sparse_waft_forward(self, sample, disp_init=None):
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
        prev_disp = disp
        for itr in range(self.iters):
            disp_in = disp.detach()

            # ---- 决定本轮要处理哪些 token ----
            keep_idx = None
            if KEEP_RATIO < 1.0 and (itr + 1) >= MIN_ITER:
                d = (disp_in - prev_disp.detach()).abs()          # (B,1,H',W')
                B_, _, Hp, Wp = d.shape
                gh, gw = Hp // 8, Wp // 8
                tok = d[:, 0, :gh*8, :gw*8].reshape(B_, gh, 8, gw, 8).mean(dim=(2, 4)).flatten(1)
                k = max(1, int(round(tok.shape[1] * KEEP_RATIO)))
                keep_idx = tok.topk(k, dim=1).indices

            warped = waft_mod.disp_warp(fmap2, disp_in, padding_mode="zeros")
            net = self.delta_proj(torch.cat([fmap1, warped, net, disp_in], dim=1))
            net = self.delta_decoder(net, keep_idx)
            info = self.delta_dist_head(net)
            delta_disp = self.delta_disp_head(net)
            mask = .25 * self.delta_mask_head(net)
            prev_disp = disp_in
            disp = disp_in + delta_disp
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
        return output

    waft_mod.WAFT.forward = sparse_waft_forward

    print(f"[P1patch] KEEP_RATIO={KEEP_RATIO} MIN_ITER={MIN_ITER} SKIP_MODE={SKIP_MODE}", flush=True)


try:
    _patch()
except Exception as e:
    import traceback
    print(f"[P1patch] 注入失败: {e}")
    traceback.print_exc()
