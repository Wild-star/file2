# 仓库推送与维护指南（GitHub）

> 本文档记录本项目（双目立体匹配融合方案 + WAFT-Stereo）推送到远程仓库的完整步骤、
> 仓库结构、大文件与第三方代码策略、以及后续维护流程。目标远程为 **GitHub**
> `https://github.com/Wild-star/file2.git`（分支 `main`）。

## 0. 本次推送结果摘要

| 项 | 值 |
|---|---|
| 远程仓库 | `https://github.com/Wild-star/file2.git` |
| 分支 | `main` |
| 提交 | `012501e` — feat: 双目立体匹配融合方案 POC + ARS 学术研究技能包部署 |
| 变更规模 | 2684 files changed, +677819 insertions |
| 同步状态 | 本地与远程一致（`origin/main..main` 为空） |
| 推送前记录 | ARS 源码版本 `a1819d8`（v3.22.0） |

推送命令：`git push origin main`（成功，输出 `c534320..012501e main -> main`）。

## 1. 前置条件

- 已安装 `git`（`git --version`）。
- 有 GitHub 账号，并对目标仓库 `Wild-star/file2` 有推送权限。
- 配置了 git 身份：
  ```bash
  git config --global user.name  "qianhongchang"
  git config --global user.email "qianhongchang@eacon.com"
  ```

## 2. 一次性初始化（新机器从零开始）

```bash
# 方式 A：直接克隆（已推送后，新机器推荐）
git clone https://github.com/Wild-star/file2.git
cd file2

# 方式 B：已有本地目录，补充远程后推送
cd /path/to/your/file2
git init
git remote add origin https://github.com/Wild-star/file2.git
git add -A && git commit -m "init"
git push -u origin main
```

## 3. 日常提交流程

```bash
cd /path/to/file2
git status                 # 查看改动
git add -A                 # 暂存全部（.gitignore 会自动排除大数据/权重）
git commit -m "feat: 描述本次改动"
git push origin main
```

## 4. 仓库结构

```
file2/
├── algorithms/            # 立体匹配算法
├── assets/                # 示例图像（Middlebury: Bottles/PianoL/Vintage/Keyboard 等，已提交）
├── bridgedepth/           # BridgeDepth 相关
├── configs/               # yacs 配置（含 eval/eth3d_S*.yaml 评估配置）
├── demo/                  # demo 入口
├── model/                 # 模型定义
├── docs/                  # 文档
│   ├── survey_stereo_flow.md          # 立体匹配/光流论文调研
│   ├── paper_design_fusionwarp.md     # FusionWarp-Stereo 论文设计（含实测消融 §5.1）
│   └── ARS_DEPLOYMENT.md              # ARS 部署与平台适配说明
├── scripts/
│   └── deploy_ars.sh                  # 跨平台 ARS 部署脚本
├── .claude/               # Claude Code 项目级激活（ARS）
│   ├── CLAUDE.md
│   ├── skills/            # deep-research / academic-paper / academic-paper-reviewer / academic-pipeline
│   └── commands/          # 16 个 /ars-* 命令
├── thirdparty/
│   └── academic-research-skills/      # ARS 完整源码（v3.22.0 @ a1819d8）
├── step1_fusion.py / step1_train.py   # 前置：窄带代价体锚融合
├── step2_fusion_composite.py          # 复合脚本（主交付，torch+numpy 自包含）
├── step3_ablation.py                  # 同条件消融对比（corr/gev-full3d/gev-sep3d）
├── p0_distill.py / p0_token_sparsity.py  # 前置诊断
├── main.py / profiler.py / submission.py / visualize.py / view_dataset.py  # WAFT 原有
└── README.md / requirements.txt / LICENSE
```

## 5. `.gitignore` 与大文件策略

已提交的 `.gitignore` 排除了以下内容（**不会**进入仓库）：

| 模式 | 排除内容 | 原因 |
|---|---|---|
| `datasets*` | `datasets/`（3.0G） | 训练数据，不入库 |
| `*ckpts*` | `ckpts/`（714M） | 中间权重 |
| `*.pth` | `p0_distill_student.pth`(295M)、`step1_joint.pth`(295M)、`step1_anchor.pth` | 模型权重，不入库 |
| `checkpoints/`、`wandb/`、`runs/`、`vis/` | 训练产物 | 可再生成 |
| `__pycache__/`、`*.py[cod]`、`*.so`、`build/`、`dist/`、`*.egg-info/` | 构建产物 | 可再生成 |

已提交的大文件：`assets/Concat.gif`（79M，< 100M 单文件上限）。若日后单文件超过
100M，需改用 **Git LFS**（见 §8）。

## 6. 第三方代码（`thirdparty/academic-research-skills/`）说明

- 来源：`https://github.com/Imbad0202/academic-research-skills`，v3.22.0，commit `a1819d8`。
- 许可证：**CC BY-NC 4.0**（非商业）。已保留 `LICENSE`、`NOTICE.md`、`THIRD_PARTY.md`、`CITATION.cff`。
- **嵌套 git 已去除**：提交前删除了 `thirdparty/academic-research-skills/.git`，使内容作为
  普通文件入库（避免被 git 识别为 gitlink/submodule）。版本号已记录在上表。

更新第三方源码：

```bash
# 重新克隆最新版并覆盖（或临时克隆后替换目录）
git clone --depth 1 https://github.com/Imbad0202/academic-research-skills.git /tmp/ars
# 记录版本
git -C /tmp/ars rev-parse HEAD
# 替换并去掉 .git
rm -rf thirdparty/academic-research-skills
mv /tmp/ars thirdparty/academic-research-skills
rm -rf thirdparty/academic-research-skills/.git
# 重新部署到 .claude/（见 ARS_DEPLOYMENT.md）
bash scripts/deploy_ars.sh
git add -A && git commit -m "chore: 更新 ARS 至 <version>" && git push origin main
```

## 7. 认证与凭据

- 当前使用 **HTTPS + credential store**（`git config credential.helper` = `store`）。
- 首次推送若提示输入用户名/密码，密码处粘贴 **Personal Access Token (PAT)**：
  GitHub → Settings → Developer settings → Personal access tokens → 勾选 `repo` 权限。
- 改用 SSH（可选）：
  ```bash
  ssh-keygen -t ed25519 -C "you@example.com"      # 已有 key 可跳过
  cat ~/.ssh/id_ed25519.pub                        # 添加到 GitHub → Settings → SSH keys
  git remote set-url origin git@github.com:Wild-star/file2.git
  ```

## 8. 常见问题

| 现象 | 说明 / 处理 |
|---|---|
| `fatal: 无法在 1000 ms 获得凭证存储锁: 只读文件系统` | 凭据存储锁写入失败（只读文件系统/沙箱），**不影响 push 本身**；可忽略，或用 SSH 规避 |
| 单文件 > 100M 被拒 | GitHub 单文件上限 100M；改用 Git LFS：`git lfs track "*.pth"` 等 |
| `warning: adding embedded git repository` | 目录内含 `.git`；删除内层 `.git`（如 §6）再 `git add` |
| push 前本地落后远程 | `git pull --rebase origin main` 后重推 |
| 想撤销最近一次提交 | `git reset --soft HEAD~1`（保留改动，仅撤提交） |

## 9. 相关文档

- `docs/ARS_DEPLOYMENT.md` — ARS 部署与跨平台（Claude Code / DSH / Linux / macOS / Windows）适配。
- `docs/paper_design_fusionwarp.md` — FusionWarp-Stereo 论文设计与消融结论。
- `docs/survey_stereo_flow.md` — 近五年立体匹配/光流论文调研。
- `README.md` — 项目原有说明。
