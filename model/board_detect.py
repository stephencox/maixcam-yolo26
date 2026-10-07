"""On-board detections for the board image list. Usage on the board:
python3 board_detect.py <model.mud> <image dir> <conf> | sed -n 's/^Y26JSON //p' > out.json
MaixPy logs to stdout on import, so the JSON goes on one line behind a marker."""
import json
import os
import sys
from maix import image, nn

det = nn.YOLO26(model=sys.argv[1], dual_buff=False)
conf = float(sys.argv[3])
res = {}
for f in sorted(os.listdir(sys.argv[2])):
    if not f.endswith(".jpg"):
        continue
    img = image.load(os.path.join(sys.argv[2], f), format=image.Format.FMT_RGB888)
    objs = det.detect(img, conf_th=conf, iou_th=0.45, fit=image.Fit.FIT_CONTAIN)
    res[f] = [[o.x, o.y, o.w, o.h, o.class_id, o.score] for o in objs]
print("Y26JSON " + json.dumps(res), flush=True)
