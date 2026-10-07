import json
from eval_map import COCO80_TO_91, map50

def test_coco_mapping():
    assert len(COCO80_TO_91) == 80 and COCO80_TO_91[0] == 1 and COCO80_TO_91[15] == 17 and COCO80_TO_91[79] == 90

def test_perfect_and_empty(tmp_path):
    ann = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 17, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}],
        "categories": [{"id": c} for c in COCO80_TO_91],
    }
    p = tmp_path / "ann.json"; p.write_text(json.dumps(ann))
    assert map50({1: [(10, 10, 30, 30, 15, 0.9)]}, str(p), [1]) > 0.99
    assert map50({1: []}, str(p), [1]) == 0.0
