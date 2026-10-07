from select_images import select

NAMES = [f"{i:012d}.jpg" for i in range(5000)]

def test_sizes_and_disjoint():
    s = select(NAMES, seed=20261007)
    assert (len(s["calib"]), len(s["eval"]), len(s["board"])) == (500, 50, 10)
    a, b, c = map(set, (s["calib"], s["eval"], s["board"]))
    assert not (a & b or a & c or b & c)

def test_deterministic_and_order_independent():
    assert select(NAMES, seed=20261007) == select(list(reversed(NAMES)), seed=20261007)

def test_seed_changes_selection():
    assert select(NAMES, seed=1)["eval"] != select(NAMES, seed=2)["eval"]
