"""Python mirror of maix::nn::YOLO26 post-processing (MaixCDK v4.12.5,
components/nn/include/maix_nn_yolo26.hpp: _generate_proposals, _correct_bbox).
Used to score FP32 (onnxruntime) and INT8 (TPU-MLIR simulator) outputs the
same way the board does. Keep in lockstep with that header."""
import cv2
import numpy as np

LOGIT_THRESHOLD = -0.2

def letterbox(img_rgb: np.ndarray, size: int):
    """Mirror of the board's FIT_CONTAIN preprocessing: NN::forward_image calls
    Image::resize(w, h, fit) whose default method is NEAREST (not bilinear);
    verified byte-identical against a MaixCAM. Returns (x CHW float32 0..1,
    scale, pad_x, pad_y)."""
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h)
    # Board: NN::forward_image calls Image::resize(w, h, fit) with the default
    # ResizeMethod::NEAREST; FIT_CONTAIN then does cv::resize(img, tmp,
    # cv::Size(), scale, scale, INTER_NEAREST) with a float scale, so the
    # output size follows from fx/fy. Verified byte-identical on a MaixCAM.
    f = float(np.float32(scale))
    resized = cv2.resize(img_rgb, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    nh, nw = resized.shape[:2]
    px, py = (size - nw) / 2.0, (size - nh) / 2.0
    canvas = np.zeros((size, size, 3), np.uint8)
    x0, y0 = int(px), int(py)
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    x = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return x, float(scale), float(px), float(py)

def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

def _clamp_bbox(x, y, w, h, img_w, img_h):
    """Mirror of _clamp_bbox (clamp to the original image), on ints."""
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    return x, y, w, h

def decode(outputs, names, size, conf, img_w, img_h):
    """Detections as (x, y, w, h, class_id, score) with int boxes in image
    pixels. Arithmetic is float32 like the C++, and boxes are truncated toward
    zero (C float->int) where nn::Object's int fields store them: after the
    input-space proposal and again in _correct_bbox before _clamp_bbox."""
    f32 = np.float32
    dets = []
    for bname, cname in zip(names["bbox"], names["cls"]):
        b = outputs[bname][0]            # 4,H,W
        c = outputs[cname][0]            # C,H,W
        fh, fw = c.shape[1:]
        sf = f32(size // fw)             # int stride, as float
        max_logit = c.max(axis=0)
        cls_id = c.argmax(axis=0)
        ys, xs = np.nonzero(max_logit >= LOGIT_THRESHOLD)
        for ay, ax in zip(ys, xs):
            score = float(_sigmoid(max_logit[ay, ax]))
            if score <= conf:
                continue
            l, t, r, btm = (f32(v) for v in b[:, ay, ax])
            cx, cy = (f32(ax) + f32(0.5)) * sf, (f32(ay) + f32(0.5)) * sf
            x, y = max(f32(0), cx - l * sf), max(f32(0), cy - t * sf)
            w = min((l + r) * sf, f32(size) - x)
            h = min((t + btm) * sf, f32(size) - y)
            if w > 0 and h > 0:
                dets.append((int(x), int(y), int(w), int(h), int(cls_id[ay, ax]), score))
    if img_w == size and img_h == size:     # _correct_bbox early-returns
        return dets
    scale = min(f32(size) / f32(img_w), f32(size) / f32(img_h))
    pad_w = (f32(size) - f32(img_w) * scale) / f32(2)
    pad_h = (f32(size) - f32(img_h) * scale) / f32(2)
    out = []
    for x, y, w, h, k, s in dets:
        x, y = int((f32(x) - pad_w) / scale), int((f32(y) - pad_h) / scale)
        w, h = int(f32(w) / scale), int(f32(h) / scale)
        out.append((*_clamp_bbox(x, y, w, h, img_w, img_h), k, s))
    return out
