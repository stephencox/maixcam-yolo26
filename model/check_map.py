"""FP32 vs simulator INT8 mAP50 gate. Usage (box venv): python check_map.py 320|640"""
import json
import os
import sys

from eval_map import fp32_map50, map50

size = int(sys.argv[1])
coco = os.path.expanduser("~/coco")
d = json.load(open(f"work/{size}/sim_eval.json"))
i8 = map50({int(k[:-4]): [tuple(v) for v in vs] for k, vs in d.items()},
           f"{coco}/annotations/instances_val2017.json", [int(k[:-4]) for k in d])
f = fp32_map50(size, coco)
print(f"fp32 {f:.4f} int8 {i8:.4f} drop {100 * (f - i8):.2f} pts")
open(f"work/{size}/sim_eval_map50.txt", "w").write(f"{f:.4f} {i8:.4f}\n")
assert f - i8 <= 0.03, "INT8 drop > 3 points"
