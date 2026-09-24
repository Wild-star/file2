#!/usr/bin/env bash
# deploy_ars.sh — 把 thirdparty/academic-research-skills 激活到项目 .claude/（跨平台、幂等）
#
# 平台差异处理：
#   - 用 `cp -R` 而非 symlink（Windows 无需管理员/开发者模式，文件系统可移植）
#   - 检测 uname，给出 Windows 专属提示
#   - 只清理 ars-* 前缀命令，不误删 .claude/commands/ 下其他命令
#
# 用法：
#   bash scripts/deploy_ars.sh            # 部署
#   bash scripts/deploy_ars.sh --dry-run  # 仅检测平台并列出将复制的 skill
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$PROJECT_ROOT/thirdparty/academic-research-skills"
DEST="$PROJECT_ROOT/.claude"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

SKILLS="deep-research academic-paper academic-paper-reviewer academic-pipeline"

# ---- 平台检测 ----
case "$(uname -s)" in
  Linux*)                                   OS="linux"   ;;
  Darwin*)                                  OS="macos"   ;;
  MINGW*|MSYS*|CYGWIN*)                     OS="windows" ;;
  *)                                        OS="unknown" ;;
esac
echo "[ARS部署] 平台: $OS ($(uname -s))"

# ---- 前置检查 ----
if [ ! -d "$SRC" ]; then
  echo "[ARS部署] 缺少源码目录: $SRC" >&2
  echo "[ARS部署] 请先克隆: git clone --depth 1 https://github.com/Imbad0202/academic-research-skills.git \"$SRC\"" >&2
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "[ARS部署] (dry-run) 将复制以下 skill 到 .claude/skills/:"
  for s in $SKILLS; do echo "  - $s  (含 shared/)"; done
  echo "[ARS部署] (dry-run) 将复制 commands/*.md 与 .claude/CLAUDE.md"
  exit 0
fi

# ---- 部署 skills ----
mkdir -p "$DEST/skills"
for s in $SKILLS; do
  rm -rf "$DEST/skills/$s"
  cp -R "$SRC/$s" "$DEST/skills/$s"
  # SKILL.md 内 `shared/xxx` 是相对引用；把 shared/ 复制进每个 skill 目录使其可解析
  rm -rf "$DEST/skills/$s/shared"
  cp -R "$SRC/shared" "$DEST/skills/$s/shared"
  echo "  [+] skills/$s  (含 shared/)"
done

# ---- 部署 commands（只清理 ars-*，不误删其他命令）----
mkdir -p "$DEST/commands"
rm -f "$DEST"/commands/ars-*.md
cp "$SRC"/commands/*.md "$DEST/commands/"
echo "  [+] commands/ ($(ls "$DEST/commands" | wc -l | tr -d ' ') 个 ars-* 命令)"

# ---- 部署 CLAUDE.md（官方 Method 1 要求 merge）----
cp "$SRC/.claude/CLAUDE.md" "$DEST/CLAUDE.md"
echo "  [+] CLAUDE.md"

echo "[ARS部署] 完成。"

# ---- 平台提示 ----
case "$OS" in
  windows)
    echo "[提示] Windows: 若需 hooks 需装 Git Bash；python3 需真实解释器（见 docs/ARS_DEPLOYMENT.md §3.3）。"
    echo "[提示] commands 内 \${CLAUDE_PLUGIN_ROOT} 在项目级下不解析，需 Method 0 plugin 或手动改路径（§5）。"
    ;;
  macos|linux)
    echo "[提示] commands 内 \${CLAUDE_PLUGIN_ROOT} 在项目级下不解析，需 Method 0 plugin 或手动改路径（§5）。"
    ;;
esac
