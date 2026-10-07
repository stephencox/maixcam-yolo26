"""yolo26n.pt -> ONNX at <size> -> cut at the six one2one head convs.
The e2e NMS-free selection block (TopK/GatherElements/Mod) does not compile
for cv181x; maix::nn::YOLO26 decodes the six head outputs itself."""
import argparse
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path

import onnx
from onnx.utils import Extractor
from ultralytics import YOLO

HERE = Path(__file__).parent
PAT = re.compile(r"/model\.\d+/one2one_cv([23])\.(\d)/one2one_cv\1\.\2\.2/Conv_output_0$")

def head_outputs(model: onnx.ModelProto) -> dict:
    found = {}
    for node in model.graph.node:
        for o in node.output:
            m = PAT.search(o)
            if m:
                found[("bbox" if m.group(1) == "2" else "cls", int(m.group(2)))] = o
    names = {"bbox": [found[("bbox", i)] for i in range(3)], "cls": [found[("cls", i)] for i in range(3)]}
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, choices=[320, 640], required=True)
    a = ap.parse_args()
    src = tomllib.loads((HERE / "sources.toml").read_text())
    work = HERE / "work" / str(a.size)
    work.mkdir(parents=True, exist_ok=True)
    pt = HERE / "work" / "yolo26n.pt"
    if not pt.exists():
        YOLO("yolo26n.pt")                      # ultralytics downloads into cwd
        shutil.move("yolo26n.pt", pt)
    digest = hashlib.sha256(pt.read_bytes()).hexdigest()
    if digest != src["yolo26n"]["sha256"]:
        raise SystemExit(f"yolo26n.pt sha256 {digest} != pinned {src['yolo26n']['sha256']}")
    full = Path(YOLO(str(pt)).export(format="onnx", imgsz=a.size, end2end=True, opset=src["export"]["opset"], simplify=True))
    shutil.move(full, work / f"yolo26n_{a.size}_full.onnx")
    m = onnx.load(work / f"yolo26n_{a.size}_full.onnx")
    names = head_outputs(m)
    cut = Extractor(m).extract_model(["images"], names["bbox"] + names["cls"])
    onnx.save(cut, work / f"yolo26n_{a.size}_cut.onnx")
    (work / "output_names.json").write_text(json.dumps(names, indent=2))
    print(json.dumps(names, indent=2))

if __name__ == "__main__":
    main()
