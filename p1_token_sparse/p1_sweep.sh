#!/usr/bin/env bash
# P1 Token 稀疏迭代扫描：精度 + 耗时
set -e
cd ~/桌面/file2
export MAMBA_ROOT_PREFIX="$HOME/.local/share/micromamba"
MM=~/.local/bin/micromamba
export PYTHONPATH="$PWD" HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=datasets/ETH3D
TRAIN_SCENES=$(python3 -c "import json;print(' '.join(json.load(open('p0_split.json'))['train']))")

check() {
  local a=$(ls $D/two_view_training/*/im0.png 2>/dev/null | wc -l)
  local b=$(ls $D/two_view_training_gt/*/disp0GT.pfm 2>/dev/null | wc -l)
  echo "    配对校验: im0=$a gt=$b $([ "$a" = "$b" ] && echo OK || echo MISMATCH)"
}
echo "[1/3] 移出训练场景"
mkdir -p /tmp/p1_img /tmp/p1_gt
for s in $TRAIN_SCENES; do
  [ -d "$D/two_view_training/$s" ] && mv "$D/two_view_training/$s" /tmp/p1_img/
  [ -d "$D/two_view_training_gt/$s" ] && mv "$D/two_view_training_gt/$s" /tmp/p1_gt/
done
check

restore() {
  echo "[3/3] 恢复"
  for s in $TRAIN_SCENES; do
    [ -d "/tmp/p1_img/$s" ] && mv "/tmp/p1_img/$s" "$D/two_view_training/"
    [ -d "/tmp/p1_gt/$s" ] && mv "/tmp/p1_gt/$s" "$D/two_view_training_gt/"
  done
  check
}
trap restore EXIT

echo
echo "[2/3] 稀疏度扫描（官方评测协议，留出 14 场景）"
printf "  %-12s %-30s %-30s %s\n" "KEEP_RATIO" "nonocc(epe,d1,bad1.0)" "all(epe,d1,bad1.0)" "推理耗时"
echo "  --------------------------------------------------------------------------------------------"
for r in 1.0 0.7 0.5 0.3 0.2; do
  tag="r${r}"
  PYTHONPATH="/tmp/p1patch:$PWD" P1_KEEP_RATIO=$r P1_MIN_ITER=2 \
    timeout 900 $MM run -n waft-stereo python main.py --num-gpus 1 --eval-only \
      --config-file configs/eval/eth3d_S.yaml --ckpt ckpts/SynLarge/DAv2S-4.pth \
      > "/tmp/p1_$tag.log" 2>&1 || true
  r1=$(grep -A3 "results for eth3d_nonocc" "/tmp/p1_$tag.log" | grep "copypaste: 0" | sed 's/.*copypaste: //')
  r2=$(grep -A3 "results for eth3d_all" "/tmp/p1_$tag.log" | grep "copypaste: 0" | sed 's/.*copypaste: //')
  tt=$(grep -oE "Total inference time: [0-9:.]+" "/tmp/p1_$tag.log" | head -1 | sed 's/.*time: //')
  printf "  %-12s %-30s %-30s %s\n" "$r" "${r1:-失败}" "${r2:-失败}" "${tt:-?}"
done
