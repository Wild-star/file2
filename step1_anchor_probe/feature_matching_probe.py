import os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/home/qianhongchang@eacon.com/桌面/file2')
from PIL import Image
from bridgedepth.config import get_cfg
from algorithms.waft import WAFT
from model.utils import Padder, disp_warp
from bridgedepth.utils.frame_utils import readDispETH3D
cfg = get_cfg(); cfg.merge_from_file('/home/qianhongchang@eacon.com/桌面/file2/configs/SynLarge/DAv2S-4.yaml')
m = WAFT(cfg).eval().cuda()
raw = torch.load('/home/qianhongchang@eacon.com/桌面/file2/ckpts/SynLarge/DAv2S-4.pth', map_location='cpu', weights_only=False)
m.load_state_dict({k.replace('module.',''):v for k,v in raw['model'].items()}, strict=False)

for scen in ['delivery_area_1l','delivery_area_1s','playground_1l']:
    d=f'/home/qianhongchang@eacon.com/桌面/file2/datasets/ETH3D/two_view_training/{scen}'
    a=np.asarray(Image.open(d+'/im0.png').convert('RGB')); b=np.asarray(Image.open(d+'/im1.png').convert('RGB'))
    t0=torch.from_numpy(a).permute(2,0,1).float()[None].cuda(); t1=torch.from_numpy(b).permute(2,0,1).float()[None].cuda()
    gt_np, valid_np = readDispETH3D(f'/home/qianhongchang@eacon.com/桌面/file2/datasets/ETH3D/two_view_training_gt/{scen}/disp0GT.pfm')
    gt = torch.from_numpy(gt_np)[None,None].cuda(); vt = torch.from_numpy(valid_np)[None,None].cuda()
    with torch.no_grad():
        im1 = m.normalize_image(t0); im2 = m.normalize_image(t1)
        pad = Padder(im1.shape, factor=m.factor)
        f1, f2, _ = m.encoder(torch.stack([pad.pad(im1), pad.pad(im2)], dim=1))
    # GT 降到 fmap 分辨率
    gt_lr = F.interpolate(gt, size=f1.shape[-2:], mode='nearest') * 0.5
    v_lr  = F.interpolate(vt.float(), size=f1.shape[-2:], mode='nearest') > 0.5
    R = 6
    sims = []
    for o in range(-R, R+1):
        w = disp_warp(f2, gt_lr + o, padding_mode='zeros')
        sims.append((F.normalize(f1,dim=1)*F.normalize(w,dim=1)).sum(1,keepdim=True))
    sim = torch.cat(sims,1)                       # (1,2R+1,h,w)
    # 只在有效像素上统计
    vm = v_lr[0,0].flatten()
    S = sim[0].permute(1,2,0).reshape(-1, 2*R+1)[vm]      # (N, 2R+1)
    argmax = S.argmax(1)
    center = R
    within1 = ((argmax - center).abs() <= 1).float().mean().item()*100
    within2 = ((argmax - center).abs() <= 2).float().mean().item()*100
    # 峰锐度：中心值 vs 平均值
    c = S[:, center]
    ratio = (c / (S.mean(1)+1e-6)).mean().item()
    print(f"  {scen:18s} 有效像素 {S.shape[0]:6d} | argmax 落在正确位置±1: {within1:5.1f}%  ±2: {within2:5.1f}% | 中心/均值 = {ratio:.3f}")
