import os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/home/qianhongchang@eacon.com/桌面/file2')
from PIL import Image
from bridgedepth.config import get_cfg
from step1_fusion import WAFTAnchor, load_pretrained, narrow_band_cost
from model.utils import Padder
from bridgedepth.utils.frame_utils import readDispETH3D
cfg = get_cfg(); cfg.merge_from_file('/home/qianhongchang@eacon.com/桌面/file2/configs/SynLarge/DAv2S-4.yaml')
m = WAFTAnchor(cfg, R=4).eval().cuda()
load_pretrained(m)
ck = torch.load('/home/qianhongchang@eacon.com/桌面/file2/step1_anchor.pth', map_location='cpu', weights_only=False)
m.load_state_dict(ck['anchor'], strict=False)
print("已载入训练后的匹配分支")

def probe(scen, R=4):
    d=f'/home/qianhongchang@eacon.com/桌面/file2/datasets/ETH3D/two_view_training/{scen}'
    a=np.asarray(Image.open(d+'/im0.png').convert('RGB')); b=np.asarray(Image.open(d+'/im1.png').convert('RGB'))
    t0=torch.from_numpy(a).permute(2,0,1).float()[None].cuda(); t1=torch.from_numpy(b).permute(2,0,1).float()[None].cuda()
    gt_np, valid_np = readDispETH3D(f'/home/qianhongchang@eacon.com/桌面/file2/datasets/ETH3D/two_view_training_gt/{scen}/disp0GT.pfm')
    gt = torch.from_numpy(gt_np)[None,None].cuda(); vt = torch.from_numpy(valid_np)[None,None].cuda()
    with torch.no_grad():
        im1 = m.normalize_image(t0); im2 = m.normalize_image(t1)
        pad = Padder(im1.shape, factor=m.factor)
        i1p, i2p = pad.pad(im1), pad.pad(im2)
        m1 = m.mb(i1p); m2 = m.mb(i2p)                     # 训练后的匹配特征（1/8 分辨率）
    gt_m = F.interpolate(gt, size=m1.shape[-2:], mode='nearest') * 0.125   # 1/8 分辨率
    v_m  = (F.interpolate(vt.float(), size=m1.shape[-2:], mode='nearest') > 0.5)[0,0].flatten()
    cost = narrow_band_cost(m1, m2, gt_m, R)               # (1,2R+1,h,w)
    S = cost[0].permute(1,2,0).reshape(-1, 2*R+1)[v_m]
    am = S.argmax(1); center = R
    w1 = ((am-center).abs()<=1).float().mean().item()*100
    w2 = ((am-center).abs()<=2).float().mean().item()*100
    return w1, w2, S.shape[0]

print("\n=== 训练后匹配分支的信号质量（GT 对齐，1/8 分辨率）===")
print(f"{'场景':<20}{'有效像素':>10}{'argmax ±1':>12}{'argmax ±2':>12}")
for scen in ['delivery_area_1l','delivery_area_1s','playground_1l']:
    w1,w2,n = probe(scen)
    print(f"{scen:<20}{n:>10d}{w1:>11.1f}%{w2:>11.1f}%")
print("\n【对照】未训练的 DAv2 特征（Step -1 实测）: ±1 仅 46.2% / 48.0% / 54.7%")
