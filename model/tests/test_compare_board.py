from compare_board import compare, iou

def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0

def test_pass_with_reordered_dets():
    sim = {"a.jpg": [[0, 0, 10, 10, 15, 0.9], [50, 50, 10, 10, 0, 0.8]]}
    board = {"a.jpg": [[50, 50, 10, 10.2, 0, 0.79], [0, 0, 10, 10, 15, 0.91]]}
    assert compare(sim, board) == []

def test_fail_count_class_iou_and_missing_image():
    sim = {"a.jpg": [[0, 0, 10, 10, 15, 0.9]], "b.jpg": [[0, 0, 10, 10, 1, 0.9]], "c.jpg": []}
    board = {"a.jpg": [[0, 0, 10, 10, 16, 0.9]], "b.jpg": [[0, 0, 5, 5, 1, 0.9]]}
    errs = compare(sim, board)
    assert any("a.jpg" in e for e in errs)
    assert any("b.jpg" in e for e in errs)
    assert any("c.jpg" in e for e in errs)
