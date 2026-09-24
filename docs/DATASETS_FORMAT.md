# 数据集存储格式规范

本文档规定 WAFT-Stereo / FusionWarp 各数据集在 `datasets/` 目录下**应当**如何组织。
数据加载代码见 `bridgedepth/dataloader/datasets.py`（目录结构）与
`bridgedepth/utils/frame_utils.py`（视差读取规则）。

---

## 1. 总览

- **统一根目录**：所有数据集放在项目根的 `datasets/` 下（`.gitignore` 已排除）。
- **路径硬编码**：每个数据集类的 `root` 默认值写死在代码里（如 `root='datasets/ETH3D'`），
  `config` 里的 `DATASETS.TRAIN` 只用**名字**选择数据集，不改路径。
- **统一接口**：无论哪个数据集，`__getitem__` 都归一化返回：

  | 字段 | 类型 | 含义 |
  |---|---|---|
  | `img1` / `img2` | `Tensor (3,H,W)` | 左右图（RGB） |
  | `disp` | `Tensor (H,W)` | 左图视差（单位：像素） |
  | `valid` | `Tensor (H,W)` | 有效 mask（`disp>0 & disp<1e3` 或数据集特定规则） |

- **图片命名约定**：左右图是两张独立图片；视差是左图像素相对右图的水平位移。
- **分辨率**：多数数据集支持 `F/H/Q`（全/半/四分之一）降采样（`skip=1/2/4`）。

---

## 2. 各数据集目录结构 + 视差格式

### SceneFlow — `datasets/sceneflow/`
```
sceneflow/
├── FlyingThings3D/{frames_finalpass|frames_cleanpass|disparity}/{TRAIN|TEST}/<pass>/<seq>/left/*.png
├── Monkaa/{frames_finalpass|frames_cleanpass|disparity}/<scene>/left/*.png
└── driving/{frames_finalpass|frames_cleanpass|disparity}/<fps>/<scene>/<cam>/left/*.png
```
- 左右图：`left/*.png` 与 `right/*.png`（同名）。
- 视差：`.pfm`（`disparity` 目录，与 left 对应，扩展名 `.pfm`）。
- 读取：`readPFM`。

### KITTI — `datasets/KITTI/`
```
KITTI/
├── 2012/{training,testing}/colored_0|colored_1/*_10.png
├── 2012/training/disp_{occ,noc}/*_10.png
├── 2015/{training,testing}/image_2|image_3/*_10.png
└── 2015/training/disp_{occ,noc}_0/*_10.png
```
- 左右图：`colored_0/colored_1`（2012）或 `image_2/image_3`（2015）。
- 视差：`disp_occ_0` / `disp_noc_0`（occ=含遮挡，noc=非遮挡），uint16 PNG，值 `/256.0`。

### Middlebury — `datasets/middlebury/`
```
middlebury/MiddEval3/
├── trainingF/<scene>/im0.png, im1.png, disp0GT.pfm, mask0nocc.png
└── testF/<scene>/im0.png, im1.png
```
- 左右图：`im0.png` / `im1.png`。
- 视差：`disp0GT.pfm`（readPFM）；非遮挡评估需 `mask0nocc.png`（255=有效）。
- 另支持 2005/2006/2014/2021 历史 split（见 `Middlebury.__init__`）。

### ETH3D — `datasets/ETH3D/`
```
ETH3D/
├── two_view_training/<scene>/im0.png, im1.png
└── two_view_training_gt/<scene>/disp0GT.pfm, mask0nocc.png
```
- 左右图：`im0.png` / `im1.png`。
- 视差：`disp0GT.pfm`（readPFM）；非遮挡需 `mask0nocc.png`。

### SintelStereo — `datasets/SintelStereo/`
```
SintelStereo/training/
├── <scene>_left/<pass>/frame_*.png
├── <scene>_right/<pass>/frame_*.png
└── disparities/<scene>/frame_*.png
```
- 视差：RGB 三通道编码，`disp = R*4 + G/2^6 + B/2^14`。

### FallingThings — `datasets/FallingThings/`
```
FallingThings/{single|mixed}/<...>/*.left.jpg, *.right.jpg, *.left.depth.png
```
- 视差：`*.left.depth.png`（深度图），需同目录 `_camera_settings.json` 取 `fx`，
  `disp = fx*6*100/depth`。

### TartanAir — `datasets/TartanAir/`
```
TartanAir/<env>/<difficulty>/<traj>/image_left/*_left.png, image_right/*_right.png,
                                depth_left/*_left_depth.npy
```
- 视差：`*_left_depth.npy`（深度），`disp = 80/depth`。

### CREStereo — `datasets/CREStereo/`
```
CREStereo/{shapenet|reflective|tree|hole}/<scene>/*_left.jpg, *_right.jpg, *_left.disp.png
```
- 视差：`*_left.disp.png`，值 `/32.0`。

### Virtual KITTI 2 — `datasets/VKITTI2/`
```
VKITTI2/SceneXX/<variant>/frames/
├── rgb/Camera_0/rgb_*.jpg, rgb/Camera_1/rgb_*.jpg
└── depth/Camera_0/depth_*.png
```
- 视差由深度换算：`disp = 725.0087 * 0.532725 * 100 / depth`。

### Carla Highres (HR-VS) — `datasets/HR-VS/carla-highres/`
```
HR-VS/carla-highres/trainingF/<scene>/im0.png, im1.png, disp0GT.pfm
```
- 视差：`disp0GT.pfm`（readPFM）。

### InStereo2K — `datasets/InStereo2K/`
```
InStereo2K/<scene>/<idx>/left.png, right.png, left_disp.png
```
- 视差：`left_disp.png`，值 `/100.0`。

### Booster — `datasets/booster/`
```
booster/train/balanced/<scene>/camera_00/im*.png, camera_02/im*.png, disp_00.npy
```
- 视差：`disp_00.npy`（np.load，直接是视差）。

### FSD — `datasets/FSD/`
```
FSD/<env>/<scene>/<idx>/
├── left/rgb/*.jpg, right/rgb/*.jpg
└── left/disparity/*.png
```
- 视差：RGB 三通道编码深度，`disp = (R*255² + G*255 + B) / 1000`。

### Spring — `datasets/spring/`
```
spring/train_val/<scene>/frame_left/*.png, frame_right/*.png, disp1_left/*.dsp5
```
- 视差：`.dsp5`（h5py，内含 `disparity` 数据集，读取后 `[::2,::2]` 降采样）。

### TartanGround — `datasets/TartanGround/`
```
TartanGround/<scene>/<seq>/<...>/image_lcam_{front|top|bottom}/*.png,
                              image_rcam_{front|top|bottom}/*.png,
                              depth_lcam_{front|top|bottom}/*.png
```
- 视差：深度图 PNG，`disp = 80/(depth+1e-6)`；`back` 相机左右互换。

### UnrealStereo4K — `datasets/UnrealStereo4K/`
```
UnrealStereo4K/<scene>/Image0/*.png, Image1/*.png, Disp0/*.npy
```
- 视差：`Disp0/*.npy`（np.load，直接是视差）。

### WMGStereo — `datasets/WMGStereo/`
```
WMGStereo/release_subset/<category>/<scene>/<idx>/
├── Image/camera_0/*.png, Image/camera_1/*.png
└── disparity/camera_0/*.npy
```
- 视差：`disparity/camera_0/*.npy`（np.load，直接是视差）。

> ⚠️ **注意**：本机当前 `datasets/WMGStereo/` 下载的是**原始格式**（`<hash>/frames/Materials|camera_0|1/...`），
> 与上述 `release_subset` 结构不符；`indoor_0001-0010.tar.gz` 也未解压。需重组为 `release_subset` 格式后才能被加载。

---

## 3. 视差读取规则速查（`frame_utils.py`）

| 数据集 | 视差文件 | 转换 |
|---|---|---|
| KITTI | PNG uint16 | `/256.0` |
| ETH3D / Middlebury / CarlaHighres / SceneFlow | `.pfm` | `readPFM` |
| SintelStereo | PNG RGB | `R*4 + G/2^6 + B/2^14` |
| FallingThings | PNG 深度 | `fx*6*100/depth` |
| TartanAir | `.npy` 深度 | `80/depth` |
| CREStereo | PNG | `/32.0` |
| VKITTI2 | PNG 深度 | `f*baseline*100/depth` |
| InStereo2K | PNG | `/100.0` |
| Booster / UnrealStereo4K / WMGStereo | `.npy` | 直接视差 |
| FSD | PNG RGB | `(R*255²+G*255+B)/1000` |
| Spring | `.dsp5` (h5py) | `disparity[::2,::2]` |
| TartanGround | PNG 深度 | `80/(depth+1e-6)` |

---

## 4. 快速校验一个数据集

```bash
# 查看某个数据集能否加载 + 打印样本形状（在项目根、激活 .venv 后）
python view_dataset.py --dataset eth3d        # 或 kitti / middlebury / sceneflow / wmgstereo / ...
```

预期输出类似：
```
<样本数>
torch.Size([3, H, W]) torch.Size([3, H, W]) torch.Size([H, W]) torch.Size([H, W])
```
（`img1`, `img2`, `disp`, `valid` 四件套），并生成 `vis/datasets/<name>/` 可视化图。
