#!/usr/bin/env bash
# Cut ONNX -> .cvimodel for cv181x via TPU-MLIR in Docker: INT8 backbone/neck,
# BF16 detection head. All-INT8 lost 7.96 mAP50 pts at 320 (0.2846 -> 0.2051):
# the cls branch activations span up to ~+-260 (logits down to -196), so INT8
# steps are too coarse. Clipping the cls logits to +-8 (with or without forcing
# the conv threshold) did not help (0.2051 / 0.1978) because the loss is inside
# the branch. Every /model.23/ op in BF16 (--quantize_table) recovers 0.2734.
# (Numbers above used a bilinear letterbox; with the board's nearest-neighbour
# resize FP32 is 0.2690 and this build 0.2498.)
# No --fuse_preprocess: the .mud mean/scale normalise input on the board
# (nn::YOLO26::detect passes them to forward_image).
set -euo pipefail
SIZE=${1:?usage: convert.sh 320|640}
COCO=${COCO:-$HOME/coco}
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$HERE/work/$SIZE"
IMG=maixcam-yolo26/tpu_mlir:1.28.1
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -t "$IMG" - <<'EOF'
FROM sophgo/tpuc_dev:latest
RUN pip install --no-cache-dir tpu_mlir[onnx]==1.28.1
EOF
rm -rf "$W/cali" && mkdir -p "$W/cali"
while read -r f; do ln -f "$COCO/val2017/$f" "$W/cali/$f" 2>/dev/null || cp "$COCO/val2017/$f" "$W/cali/"; done < "$HERE/lists/calib.txt"
cp "$COCO/val2017/$(head -1 "$HERE/lists/eval.txt")" "$W/test_input.jpg"
OUT=$(python3 -c "import json;d=json.load(open('$W/output_names.json'));print(','.join(d['bbox']+d['cls']))")
{
docker run --rm -v "$W":/work -w /work "$IMG" bash -c "
set -e
model_transform.py --model_name yolo26n_$SIZE --model_def yolo26n_${SIZE}_cut.onnx \
  --input_shapes [[1,3,$SIZE,$SIZE]] --output_names $OUT \
  --mean 0,0,0 --scale 0.00392156862745098,0.00392156862745098,0.00392156862745098 \
  --pixel_format rgb --keep_aspect_ratio --pad_value 0 \
  --test_input test_input.jpg --test_result top_outputs.npz --mlir yolo26n_$SIZE.mlir
run_calibration.py yolo26n_$SIZE.mlir --dataset cali --input_num 500 -o cali_table
"
grep -o 'loc("/model\.23/[^"]*")' "$W/yolo26n_$SIZE.mlir" | sed 's/^loc("//; s/")$/ BF16/' | sort -u > "$W/qtable"
echo "qtable: $(wc -l < "$W/qtable") /model.23/ ops in BF16"
docker run --rm -v "$W":/work -w /work "$IMG" bash -c "
set -e
model_deploy.py --mlir yolo26n_$SIZE.mlir --quantize INT8 --calibration_table cali_table \
  --quantize_table qtable --processor cv181x --test_input yolo26n_${SIZE}_in_f32.npz --test_reference top_outputs.npz \
  --tolerance 0.85,0.45 --model yolo26n_${SIZE}_int8.cvimodel
"
} 2>&1 | tee "$W/convert.log"
ls -l "$W/yolo26n_${SIZE}_int8.cvimodel"
