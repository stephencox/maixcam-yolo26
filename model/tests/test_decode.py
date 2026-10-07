import cv2
import numpy as np
from decode import decode, letterbox

NAMES = {"bbox": ["b8", "b16", "b32"], "cls": ["c8", "c16", "c32"]}

def blank(size, nc=80):
    out = {}
    for s, b, c in zip((8, 16, 32), NAMES["bbox"], NAMES["cls"]):
        g = size // s
        out[b] = np.zeros((1, 4, g, g), np.float32)
        out[c] = np.full((1, nc, g, g), -10.0, np.float32)
    return out

def test_letterbox_square_is_plain_resize():
    img = np.full((100, 100, 3), 255, np.uint8)
    x, scale, px, py = letterbox(img, 320)
    assert x.shape == (3, 320, 320) and x.dtype == np.float32
    assert (scale, px, py) == (3.2, 0.0, 0.0)
    assert np.allclose(x, 1.0)

def test_letterbox_pads_black():
    img = np.full((160, 320, 3), 255, np.uint8)       # h=160, w=320
    x, scale, px, py = letterbox(img, 320)
    assert (scale, px, py) == (1.0, 0.0, 80.0)
    assert np.allclose(x[:, :80, :], 0.0) and np.allclose(x[:, 80:240, :], 1.0)

def test_single_cell_decode_and_letterbox_inverse():
    out = blank(320)
    # stride-32 grid is 10x10; cell (ax=3, ay=4): centre (112, 144) in input px
    out["c32"][0, 15, 4, 3] = 2.0               # class 15 (cat), sigmoid(2)=0.881
    out["b32"][0, :, 4, 3] = [1.0, 1.0, 1.0, 1.0]  # l,t,r,b in stride units
    dets = decode(out, NAMES, 320, conf=0.5, img_w=640, img_h=640)
    assert len(dets) == 1
    x, y, w, h, cid, score = dets[0]
    assert cid == 15 and abs(score - 0.8808) < 1e-3
    # input box (80,112,64,64) -> scale 0.5 back to 640x640
    assert np.allclose((x, y, w, h), (160, 224, 128, 128), atol=1e-4)

def test_logit_floor_drops_below_045_even_with_low_conf():
    out = blank(320)
    out["c32"][0, 0, 0, 0] = -0.3               # sigmoid=0.4256 < floor
    out["c16"][0, 0, 0, 0] = -0.1               # sigmoid=0.475, kept at conf 0.3
    out["b32"][0, :, 0, 0] = 1; out["b16"][0, :, 0, 0] = 1
    dets = decode(out, NAMES, 320, conf=0.3, img_w=320, img_h=320)
    assert len(dets) == 1 and abs(dets[0][5] - 0.4750) < 1e-3

def test_clamp_to_input():
    out = blank(320)
    out["c8"][0, 1, 0, 0] = 3.0
    out["b8"][0, :, 0, 0] = [5.0, 5.0, 1.0, 1.0]   # extends left/up past 0
    (x, y, w, h, *_), = decode(out, NAMES, 320, conf=0.5, img_w=320, img_h=320)
    assert x == 0 and y == 0 and w > 0 and h > 0

def test_letterbox_resizes_like_board_fit_contain():
    # Board FIT_CONTAIN: cv::resize(img, tmp, Size(), scale, scale, INTER_LINEAR)
    # with a float32 scale, so pixel sampling follows fx/fy, not nw/w.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (500, 333, 3), np.uint8)   # h=500, w=333
    x, scale, px, py = letterbox(img, 320)
    f = float(np.float32(320 / 500))
    ref = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    nh, nw = ref.shape[:2]
    assert (nh, nw) == (320, 213)
    assert (scale, px, py) == (0.64, (320 - 213) / 2.0, 0.0)
    x0 = int(px)
    got = np.rint(x.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    assert np.array_equal(got[:, x0:x0 + nw], ref)
    assert not got[:, :x0].any() and not got[:, x0 + nw:].any()

def test_non_square_decode_inverts_pad_and_clamps_to_image():
    # 640x320 image into 320: scale 0.5, content 320x160 at y 80..240 (pad_h 80).
    out = blank(320)
    out["c32"][0, 2, 2, 9] = 2.0                    # cell (ax=9, ay=2): centre (304, 80)
    out["b32"][0, :, 2, 9] = [1.0, 1.0, 1.0, 1.0]   # input box (272,48,48,64) after w clamp to 320
    out["c32"][0, 7, 7, 3] = 2.0                    # cell (ax=3, ay=7): centre (112, 240)
    out["b32"][0, :, 7, 3] = [1.0, 1.0, 1.0, 1.0]   # input box (80,208,64,64)
    dets = sorted(decode(out, NAMES, 320, conf=0.5, img_w=640, img_h=320), key=lambda d: d[4])
    assert [d[4] for d in dets] == [2, 7]
    # class 2: image (544, -64, 96, 128) -> top clamp -> (544, 0, 96, 64)
    assert np.allclose(dets[0][:4], (544, 0, 96, 64), atol=1e-4)
    # class 7: image (160, 256, 128, 128) -> bottom clamp to 320 -> (160, 256, 128, 64)
    assert np.allclose(dets[1][:4], (160, 256, 128, 64), atol=1e-4)
