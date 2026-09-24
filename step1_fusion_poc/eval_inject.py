"""
Step 1 评测注入：让 main.py 使用带锚的模型

- 把 algorithms.waft.WAFT 替换为 WAFTAnchor
- 构造时自动载入训练好的锚权重（step1_anchor.pth）
- 环境变量 ANC_ON=0/1 控制是否启用锚（用于消融对比）
"""
import os
import torch

ANC_ON = os.environ.get("ANC_ON", "1") == "1"
ANCHOR_CKPT = os.environ.get("ANCHOR_CKPT", "step1_anchor.pth")
STATS = []


def _patch():
    import algorithms.waft as waft_mod
    from step1_fusion import WAFTAnchor, load_pretrained

    _orig_init = WAFTAnchor.__init__

    def init_with_anchor(self, cfg, R=4, mch=32):
        _orig_init(self, cfg, R=R, mch=mch)
        self.use_anchor = ANC_ON
        if os.path.exists(ANCHOR_CKPT):
            ck = torch.load(ANCHOR_CKPT, map_location="cpu", weights_only=False)
            w = {}
            if isinstance(ck, dict):
                if "anchor" in ck: w.update(ck["anchor"])
                if "joint" in ck and ck.get("mode") == "joint": w.update(ck["joint"])
            else:
                w = ck
            miss = self.load_state_dict(w, strict=False)
            n = len(w) - len(miss.unexpected_keys)
            mode = ck.get("mode", "?") if isinstance(ck, dict) else "?"
            print(f"[Step1-Eval] 权重载入 {n} 枚 | mode={mode} | use_anchor={ANC_ON}", flush=True)
        else:
            print(f"[Step1-Eval] ⚠️ 未找到 {ANCHOR_CKPT}，锚为随机/零初始化", flush=True)

    WAFTAnchor.__init__ = init_with_anchor

    # 让 main.py 里的 WAFT(cfg) 构造出带锚版本
    waft_mod.WAFT = WAFTAnchor
    # main.py 用的是 from algorithms.waft import WAFT（模块级绑定）
    try:
        import main as main_mod
        if hasattr(main_mod, "WAFT"):
            main_mod.WAFT = WAFTAnchor
            print("[Step1-Eval] 已替换 main.WAFT", flush=True)
    except Exception:
        pass
    print(f"[Step1-Eval] WAFTAnchor 注入完成 (ANC_ON={ANC_ON})", flush=True)


try:
    _patch()
except Exception as e:
    import traceback
    print(f"[Step1-Eval] 注入失败: {e}")
    traceback.print_exc()
