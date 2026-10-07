# Model manifest

Regenerate with `just models` on 192.168.12.30 (see README). Any change to these
values bumps the app's minor version.

| | 320 | 640 |
|---|---|---|
| Source | yolo26n.pt sha256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | same |
| Ultralytics | `8.4.174` | same |
| ONNX opset | 17 | 17 |
| Cut outputs | bbox `/model.23/one2one_cv2.{0,1,2}/one2one_cv2.{0,1,2}.2/Conv_output_0`, cls `/model.23/one2one_cv3.{0,1,2}/one2one_cv3.{0,1,2}.2/Conv_output_0` | pending (Task 7) |
| TPU-MLIR | tpu_mlir[onnx]==1.28.1 on sophgo/tpuc_dev (built image `sha256:2509cc15a3fe394f8d20aa83f908a938863557bdd42ee1dc9d7500450c4e09a9`); mixed INT8 + BF16 head (42 `/model.23/` ops BF16 via `--quantize_table`) | pending (Task 7) |
| Calibration | 500 imgs, `lists/calib.txt` sha256 `11abd28d6ff2ea551b052322813fe7587185fc94f818eb474771aadc651547fe` | pending (Task 7) |
| FP32 mAP50 (50 eval imgs, conf 0.45) | 0.2690 (board-style nearest-neighbour letterbox) | pending (Task 7) |
| INT8 mAP50 (sim) | 0.2498 (drop 1.92 pts) | pending (Task 7) |
| Board vs sim (10 imgs) | scores and counts identical (17 dets); 14/17 boxes IoU ≥ 0.9, 3 small boxes 0.817–0.893 from the board's integer box fields; 17/17 identical with that truncation emulated | pending (Task 7) |
| Board inference ms (552×368 input, FIT_CONTAIN) | 111.3 (all-INT8 build: 109.2) | pending (Task 7) |
| .cvimodel sha256 | `e8b2855c42cf3d015212d3c4b3bcd63f4b1250ba5302def95f05edbb560fee99` (2,890,680 bytes) | pending (Task 7) |
