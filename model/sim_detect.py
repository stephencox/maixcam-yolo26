"""INT8 detections from the TPU-MLIR cmodel simulator running the deployed
.cvimodel, decoded like the board. (model_deploy deletes the intermediate
*_int8_sym_tpu.mlir weights unless --debug, which would also embed debug info
in the .cvimodel, so the simulator runs the final artifact itself.)
The runtime names outputs `<onnx name>_<OpType>_f32` (e.g. `..._Conv_f32`,
`..._clip_Clip_f32`); that suffix is stripped to match output_names.json.
Usage (in container): python3 sim_detect.py --size 320 --list lists/eval.txt --coco /coco --out work/320/sim_eval.json"""
import argparse
import glob
import json
import re
from pathlib import Path

import cv2
from tpu_mlir.python.tools.model_runner import model_inference

from decode import decode, letterbox

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--coco", required=True)
ap.add_argument("--conf", type=float, default=0.45)
ap.add_argument("--out", required=True)
a = ap.parse_args()
here = Path(__file__).parent
work = here / "work" / str(a.size)
model = glob.glob(str(work / "*_int8.cvimodel"))
assert len(model) == 1, model
names = json.loads((work / "output_names.json").read_text())
res = {}
for f in Path(a.list).read_text().split():
    img = cv2.cvtColor(cv2.imread(f"{a.coco}/val2017/{f}"), cv2.COLOR_BGR2RGB)
    x, *_ = letterbox(img, a.size)
    out = model_inference({"images": x[None]}, model[0], dump_all=False)
    out = {re.sub(r"_[A-Za-z]+_f32$", "", k): v for k, v in out.items()}
    missing = set(names["bbox"] + names["cls"]) - out.keys()
    assert not missing, (missing, sorted(out))
    res[f] = [list(map(float, d[:4])) + [int(d[4]), float(d[5])]
              for d in decode(out, names, a.size, a.conf, img.shape[1], img.shape[0])]
Path(a.out).write_text(json.dumps(res))
print(f"{len(res)} images -> {a.out}")
