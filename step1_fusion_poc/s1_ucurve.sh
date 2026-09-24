#!/usr/bin/env bash
# Step 1 决定性实验：锚能否改变迭代的 U 曲线？
#   对照：ANC_ON=0（无锚） vs ANC_ON=1（有锚，锚模块已训练）
#   迭代：1 / 2 / 3
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
  echo "    配对校验 im0=$a gt=$b $([ "$a" = "$b" ] && echo OK || echo MISMATCH)"
}
echo "[1/3] 移出训练场景"
mkdir -p /tmp/s1_img /tmp/s1_gt
for s in $TRAIN_SCENES; do
  [ -d "$D/two_view_training/$s" ] && mv "$D/two_view_training/$s" /tmp/s1_img/
  [ -d "$D/two_view_training_gt/$s" ] && mv "$D/two_view_training_gt/$s" /tmp/s1_gt/
done
check
restore() {
  echo "[3/3] 恢复"
  for s in $TRAIN_SCENES; do
    [ -d "/tmp/s1_img/$s" ] && mv "/tmp/s1_img/$s" "$D/two_view_training/"
    [ -d "/tmp/s1_gt/$s" ] && mv "/tmp/s1_gt/$s" "$D/two_view_training_gt/"
  done
  check
}
trap restore EXIT

echo
echo "[2/3] U 曲线对比（ETH3D 留出 14 场景）"
printf "  %-8s %-7s %-30s %-30s %s\n" "锚" "迭代" "nonocc(epe,d1,bad1.0)" "all(epe,d1,bad1.0)" "耗时"
echo "  -------------------------------------------------------------------------------------------"
for on in 0 1; do
  for it in 1 2 3; do
    cfg="configs/eval/eth3d_S.yaml"
    [ "$it" = "3" ] || cfg="configs/eval/eth3d_S_iters$it.yaml"
    tag="on${on}_it${it}"
    PYTHONPATH="/tmp/s1eval:$PWD" ANC_ON=$on \
      timeout 900 $MM run -n waft-stereo python main.py --num-gpus 1 --eval-only \
        --config-file "$cfg" --ckpt ckpts/SynLarge/DAv2S-4.pth > "/tmp/s1_$tag.log" 2>&1 || true
    r1=$(grep -A3 "results for eth3d_nonocc" "/tmp/s1_$tag.log" | grep "copypaste: 0" | sed 's/.*copypaste: //')
    r2=$(grep -A3 "results for eth3d_all" "/tmp/s1_$tag.log" | grep "copypaste: 0" | sed 's/.*copypaste: //')
    tt=$(grep -oE "Total inference time: [0-9:.]+" "/tmp/s1_$tag.log" | head -1 | sed 's/.*time: //')
    printf "  %-8s %-7s %-30s %-30s %s\n" "$([ "$on" = "1" ] && echo 有 || echo 无)" "$it 轮" "${r1:-失败}" "${r2:-失败}" "${tt:-?}"
  done
done
