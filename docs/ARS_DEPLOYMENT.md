# ARS（Academic Research Skills）部署与平台适配说明

> 本文件说明如何把 `Imbad0202/academic-research-skills`（v3.22.0）部署到本仓库，
> 以及**不同平台之间的差异**——这是部署时最容易踩坑、也最容易误判的部分。

## 1. 这是什么

- **Academic Research Skills (ARS)**，作者 Cheng-I Wu（[@Imbad0202](https://github.com/Imbad0202)），
  许可证 **CC BY-NC 4.0**（非商业，保留 attribution）。
- 一套 **Claude Code 原生**的学术研究技能包，含 4 个 skill：
  | Skill | 版本 | 角色 |
  |---|---|---|
  | `deep-research` | 2.12.1 | 13-agent 深度研究 / 文献综述 / 系统综述 |
  | `academic-paper` | 3.3.1 | 12-agent 论文写作（plan/outline/revision/citation-check 等 11 个 mode） |
  | `academic-paper-reviewer` | 1.11.1 | 多视角论文评审（5 席位评审团 + 魔鬼代言人） |
  | `academic-pipeline` | 3.22.0 | 全流程编排（research→write→integrity→review→revise→finalize） |
- 另有 16 个 slash commands（`/ars-plan`、`/ars-lit-review`、`/ars-reviewer`、`/ars-citation-check` 等）。

## 2. 本次部署做了什么（目录布局）

```
file2/
├── thirdparty/academic-research-skills/   # 完整源码浅克隆（保留 .git，可 pull 更新）
│   ├── LICENSE / NOTICE.md / THIRD_PARTY.md / CITATION.cff   # 许可与出处（必须保留）
│   ├── deep-research/ academic-paper/ academic-paper-reviewer/ academic-pipeline/
│   ├── shared/  scripts/  commands/  docs/  agents/  ...
├── .claude/                               # Claude Code 项目级激活目录
│   ├── CLAUDE.md                          # ARS 的路由指令（官方 Method 1 要求 merge）
│   ├── skills/
│   │   ├── deep-research/                 # 每个 skill 目录内带一份 shared/
│   │   ├── academic-paper/
│   │   ├── academic-paper-reviewer/
│   │   └── academic-pipeline/
│   └── commands/
│       └── ars-*.md                       # 16 个 slash commands
├── scripts/deploy_ars.sh                  # 跨平台（幂等）部署脚本
└── docs/ARS_DEPLOYMENT.md                 # 本文件
```

部署方式采用官方 **Method 1（项目级 skills）**：`cp`（而非 symlink）4 个 skill 到
`.claude/skills/`，保证在不同文件系统/平台上都可移植（symlink 在 Windows 需要管理员/开发者模式）。

## 3. 平台差异总览（核心）

ARS 的**机制可用性随宿主平台不同而不同**。官方把这一点总结在
`thirdparty/academic-research-skills/docs/CONTROL_AVAILABILITY.md`（机制 × 安装渠道对照，
CA-1..3）。以下是本项目部署时的实践结论。

### 3.1 Agent 宿主平台差异矩阵

| 能力/机制 | Claude Code CLI / VS Code / JetBrains | claude.ai web / API | Claude Cowork | Codex CLI | **本环境 DeepSeek Harness Web GUI** |
|---|---|---|---|---|---|
| `/plugin` 安装（Method 0） | ✅ 完整 | ❌ | ❌ | ❌（用 sibling 仓库） | ❌ |
| 项目级 `.claude/skills/`（Method 1） | ✅ | ❌ | ❌ | ❌ | ❌ |
| slash commands `/ars-*` | ✅ | ❌ | ❌ | ❌ | ❌ |
| 多 agent 子代理编排 | ✅（Task 工具） | ❌ | ❌ | ✅（自有） | ✅（自有 subagent/workflow） |
| `scripts/` 脚本后盾 | ✅ | ❌ | ❌ | 部分 | 部分（自有 bash/python） |
| hooks（写保护等） | ✅ | ❌ | ❌ | ❌ | ❌ |

### 3.2 本环境（DeepSeek Harness Web GUI）的差异 —— 最重要

ARS 是 **Claude Code 专用**的 skill 包，其加载机制依赖 Claude Code 的
`~/.claude/skills/` 或 `.claude/skills/` 目录发现 + `/plugin` 安装。

**当前 DeepSeek Harness（DSH）环境不是 Claude Code**，因此：

1. DSH 的 skill 是 **harness 内置的 session catalog**（通过 `skill` 工具按名加载，
   例如本会话里的 `research`、`paper-ingest`、`code-review`、`writing-for-agents` 等），
   **不读取项目目录下的 `.claude/skills/`**。所以本次部署进 `.claude/skills/` 的 4 个
   ARS skill **不会**在 DSH 里自动成为可加载 skill。
2. DSH 没有 `/plugin`、没有 slash command（`/ars-*`）机制，`${CLAUDE_PLUGIN_ROOT}`
   这类环境变量也不存在。
3. **在 DSH 里怎么用 ARS**：
   - 把 `thirdparty/academic-research-skills/<skill>/SKILL.md` 及其
     `references/`、`templates/`、`agents/` **当作方法论参考文档**，手动让 agent 遵循
     （例如写论文时读取 `academic-paper/SKILL.md` 的 outline/revision 规则）。
   - 或**映射到 DSH 已内置的等价 skill**：
     `research`（深研/文献）、`paper-ingest`（论文与实现对齐）、`code-review`（评审）、
     `writing-for-agents`（写 agent 文档）。
   - 用 DSH 的 `subagent` / `workflow` 机制可**部分复现** ARS 的"多 agent 角色团队"
     （如 deep-research 的 13-agent），但需要手动编排，不会像 Claude Code 里那样
     `academic-pipeline` 自动串联。

> 结论：本次部署的价值是**双重的**——① 让本仓库在任何 **Claude Code** 环境打开时
> 直接可用 ARS（`.claude/` 约定已就位）；② 在当前 DSH 环境里作为**可读、可引用的
> 方法论资源**（`thirdparty/` 完整源码 + 各 skill 的 SKILL.md）。

### 3.3 操作系统差异（Linux / macOS / Windows）

ARS 的 CI 只在 **Ubuntu** 上跑；macOS 社区可用；Windows 是 **best-effort**。

| 项 | Linux / macOS | Windows |
|---|---|---|
| 官方支持度 | 测试平台 | best-effort |
| 路径约定 | `~/.claude/skills/` | `%USERPROFILE%\.claude\skills\` |
| symlink 部署 | ✅ 可直接用 | ❌ 需管理员/开发者模式（**本项目用 `cp`，规避**） |
| `python3` | 系统自带 | 常是 **Microsoft Store 占位符**（非真解释器），需装 [python.org](https://www.python.org) 或 `winget install Python.Python.3.12` |
| hooks 的 `.sh` launcher | 直接跑 | 需 **Git Bash**（随 Git for Windows 安装）；没有 Git Bash 时 `PreToolUse` hook 每次调用报错（优雅降级、不阻塞写入） |
| 共享文件锁 | 直接 | `scripts/file_lock.py` 用 `msvcrt` 后端，共享读锁**降级为排他锁** |
| Pandoc（DOCX 输出，可选） | `brew install pandoc` / `apt install pandoc` | `choco install pandoc` / `winget` |
| tectonic + 字体（PDF，可选） | `apt install ttf-mscorefonts-installer`（Times New Roman） | 需装 tectonic，字体另装 |

## 4. 各安装渠道的能力可用性（Method 0 vs 1 摘要）

- **Method 0（plugin，完整功能）**：`/plugin marketplace add Imbad0202/academic-research-skills`
  → `/plugin install academic-research-skills`。支持 slash commands、hooks、sub-agent、
  scripts。但仅限 Claude Code CLI / VS Code / JetBrains。
- **Method 1（项目级 skills，本次采用）**：`cp` 4 个 skill 到 `.claude/skills/`。
  **核心 prompt-driven 能力可用**；但以下机制**降级/缺失**（对照 CONTROL_AVAILABILITY.md）：
  - `scripts/` 脚本后盾不在 `.claude/skills/` 内 → 脚本支撑的检查退化；
  - `hooks`、plugin 专属的 tools allowlist 不可用；
  - slash commands 里的 `${CLAUDE_PLUGIN_ROOT}` **无法解析**（见 §5）。

## 5. 已知注意点（如实记录）

1. **`shared/` 相对引用**：各 SKILL.md 正文大量引用 `shared/xxx`（如
   `shared/style_calibration_protocol.md`、`shared/references/...`）。这是相对仓库根的路径，
   项目级安装下不会自动命中。**本次部署把 `shared/` 复制进了每个 skill 目录**
   （`<skill>/shared/`），使相对引用可解析（代价是 4 份冗余，约 2.5MB/份，可接受）。
2. **`${CLAUDE_PLUGIN_ROOT}`**：16 个 `commands/ars-*.md` 里用 `${CLAUDE_PLUGIN_ROOT}`
   定位 `MODE_REGISTRY.md` 与各 SKILL.md。这是 **plugin 安装（Method 0）才有**的环境变量；
   项目级 `.claude/commands/` 下**无法解析**。要用 slash command 的完整路径解析，需：
   改用 Method 0 plugin 安装，或手动把 `${CLAUDE_PLUGIN_ROOT}` 替换为本仓库实际路径。
3. **更新**：ARS 每 1–2 周发版。更新源码后需重跑部署脚本：
   ```bash
   git -C thirdparty/academic-research-skills pull
   bash scripts/deploy_ars.sh
   ```
4. **嵌套 git**：`thirdparty/academic-research-skills/` 是独立浅克隆（含 `.git`），便于
   `git pull` 更新。若要把主项目整体提交到 git，需自行决定是否 `git submodule add` 或
   排除该目录，避免被当作嵌套 gitlink。

## 6. 重新部署 / 更新

```bash
# 一键（幂等，跨平台，cp 方式）
bash scripts/deploy_ars.sh

# 仅看帮助/平台检测
bash scripts/deploy_ars.sh --dry-run
```

脚本会自动检测平台（`uname`），用 `cp -R` 覆盖 `.claude/skills/`、`.claude/commands/`、
`.claude/CLAUDE.md`，并只清理 `ars-*` 前缀命令（不误删其他命令）。

## 7. 许可

- ARS 本体：**CC BY-NC 4.0**（非商业使用）。本项目为学术研究用途，符合许可；
  分发/商用前请自行评估。
- 已保留 `LICENSE`、`NOTICE.md`、`THIRD_PARTY.md`、`CITATION.cff` 于
  `thirdparty/academic-research-skills/`。
- 若在论文中用到 ARS 的方法论，可按 `CITATION.cff` 引用（DOI: 10.5281/zenodo.20696614）。
