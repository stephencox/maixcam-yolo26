# Model manifest

Regenerate with `just models` on 192.168.12.30 (see README). Any change to these
values bumps the app's minor version.

| | 320 | 640 |
|---|---|---|
| Source | yolo26n.pt sha256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | same |
| Ultralytics | `8.4.174` | same |
| ONNX opset | 17 | 17 |
| Cut outputs | bbox `/model.23/one2one_cv2.{0,1,2}/one2one_cv2.{0,1,2}.2/Conv_output_0`, cls `/model.23/one2one_cv3.{0,1,2}/one2one_cv3.{0,1,2}.2/Conv_output_0` | `/model.23/one2one_cv2.{0,1,2}/one2one_cv2.{0,1,2}.2/Conv_output_0` (bbox), `/model.23/one2one_cv3.{0,1,2}/one2one_cv3.{0,1,2}.2/Conv_output_0` (cls); same names as 320 |
| TPU-MLIR | tpu_mlir[onnx]==1.28.1 on sophgo/tpuc_dev (built image `sha256:2509cc15a3fe394f8d20aa83f908a938863557bdd42ee1dc9d7500450c4e09a9`) | same (image `sha256:2509cc15a3fe394f8d20aa83f908a938863557bdd42ee1dc9d7500450c4e09a9`) |
| Precision | INT8 + BF16 head (/model.23, 42 ops via `--quantize_table`) | INT8 + BF16 head (/model.23, 42 ops via `--quantize_table`) |
| Calibration | 500 imgs, `lists/calib.txt` sha256 `11abd28d6ff2ea551b052322813fe7587185fc94f818eb474771aadc651547fe` | 500 imgs, `lists/calib.txt` (same list as 320) |
| FP32 mAP50 (50 eval imgs, conf 0.45) | 0.2603 (board-exact decoder: nearest-neighbour letterbox, int boxes) | 0.3779 (board-exact decoder) |
| INT8 mAP50 (sim) | 0.2498 (drop 1.05 pts) | 0.3578 (drop 2.00 pts) |
| Board vs sim (10 imgs) | pass (17/17 detections: identical int boxes and classes, scores within 1e-7) | pass (`compare_board.py`: board matches simulator, 10 imgs) |
| Board inference ms (552×368 input, FIT_CONTAIN) | 111.3 (all-INT8 build: 109.2) | 641.4 (30 timed after 5 warm-up; board free RAM 75 -> 61 MB available with model loaded) |
| .cvimodel size | 2,890,680 bytes (all-INT8 build: 2,692,368) | 3,318,176 bytes |
| .cvimodel sha256 | `e8b2855c42cf3d015212d3c4b3bcd63f4b1250ba5302def95f05edbb560fee99` | `eda63b2f2598c0b99c86ee4bf1d15ce1f023395b9f8430f9d5c5926c970558dd` |

## Rejected alternatives (320, measured; do not retry)

- Pure INT8: mAP50 drop 7.96 pts (class-logit thresholds ±114 / ±92, branch activations to ~260).
- Clip class logits to ±8 before the cut: no change (7.96); TPU-MLIR keeps the conv's own threshold in front of the Clip.
- Clip ±8 plus forcing the class conv thresholds to 8 in the calibration table: drop 8.68 pts (the loss is inside the class branch).
- BF16 for only the three final class convs: worse (drop ~10 pts).
- BF16 for `/model.22/` + `/model.23/`: drop 0.74 pts but 3.36 MB; the `/model.23/` head alone is enough.

(Pure INT8 and the Clip variants were measured with a bilinear letterbox and float boxes; the shipped numbers above use the board-exact decoder.)
