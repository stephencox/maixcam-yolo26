"""Python mirror of maix::nn::YOLO26 post-processing (MaixCDK v4.12.5,
components/nn/include/maix_nn_yolo26.hpp: _generate_proposals, _correct_bbox).
Used to score FP32 (onnxruntime) and INT8 (TPU-MLIR simulator) outputs the
same way the board does. Keep in lockstep with that header."""
import cv2
import numpy as np

LOGIT_THRESHOLD = -0.2

def letterbox(img_rgb: np.ndarray, size: int):
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    px, py = (size - nw) / 2.0, (size - nh) / 2.0
    canvas = np.zeros((size, size, 3), np.uint8)
    x0, y0 = int(px), int(py)
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return x, float(scale), float(px), float(py)

def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

def _clamp_bbox(x, y, w, h, img_w, img_h):
    """Mirror of _clamp_bbox (clamp to the original image)."""
    if x < 0:
        w += x
        x = 0.0
    if y < 0:
        h += y
        y = 0.0
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    return x, y, w, h

def decode(outputs, names, size, conf, img_w, img_h):
    dets = []
    for bname, cname in zip(names["bbox"], names["cls"]):
        b = outputs[bname][0]            # 4,H,W
        c = outputs[cname][0]            # C,H,W
        fh, fw = c.shape[1:]
        stride = size / fw
        max_logit = c.max(axis=0)
        cls_id = c.argmax(axis=0)
        ys, xs = np.nonzero(max_logit >= LOGIT_THRESHOLD)
        for ay, ax in zip(ys, xs):
            score = float(_sigmoid(max_logit[ay, ax]))
            if score <= conf:
                continue
            l, t, r, btm = b[:, ay, ax]
            cx, cy = (ax + 0.5) * stride, (ay + 0.5) * stride
            x, y = max(0.0, cx - l * stride), max(0.0, cy - t * stride)
            w = min((l + r) * stride, size - x)
            h = min((t + btm) * stride, size - y)
            if w > 0 and h > 0:
                dets.append([x, y, w, h, int(cls_id[ay, ax]), score])
    if img_w == size and img_h == size:     # _correct_bbox early-returns
        return [tuple(d) for d in dets]
    scale = min(size / img_w, size / img_h)
    pad_w, pad_h = (size - img_w * scale) / 2.0, (size - img_h * scale) / 2.0
    out = []
    for x, y, w, h, k, s in dets:
        x, y, w, h = _clamp_bbox((x - pad_w) / scale, (y - pad_h) / scale, w / scale, h / scale, img_w, img_h)
        out.append((x, y, w, h, k, s))
    return out
