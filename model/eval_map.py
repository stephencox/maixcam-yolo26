"""COCO mAP50 over a fixed image subset, using pycocotools."""
import contextlib
import io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def map50(dets_by_image, ann_file, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann_file)
        res = [{"image_id": i, "category_id": COCO80_TO_91[k], "bbox": [x, y, w, h], "score": s}
               for i, ds in dets_by_image.items() for (x, y, w, h, k, s) in ds]
        if not res:
            return 0.0
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(image_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1])

def fp32_map50(size: int, coco_dir: str, conf: float = 0.45) -> float:
    import json as _json
    from pathlib import Path
    import cv2
    import onnxruntime as ort
    from decode import decode, letterbox
    here = Path(__file__).parent
    work = here / "work" / str(size)
    names = _json.loads((work / "output_names.json").read_text())
    sess = ort.InferenceSession(str(work / f"yolo26n_{size}_cut.onnx"), providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()]
    files = (here / "lists" / "eval.txt").read_text().split()
    dets = {}
    for f in files:
        img = cv2.cvtColor(cv2.imread(f"{coco_dir}/val2017/{f}"), cv2.COLOR_BGR2RGB)
        x, *_ = letterbox(img, size)
        res = dict(zip(outs, sess.run(outs, {"images": x[None]})))
        dets[int(f[:-4])] = decode(res, names, size, conf, img.shape[1], img.shape[0])
    return map50(dets, f"{coco_dir}/annotations/instances_val2017.json", list(dets))

if __name__ == "__main__":
    import sys
    print(f"fp32 mAP50 @{sys.argv[1]}: {fp32_map50(int(sys.argv[1]), sys.argv[2]):.4f}")
