"""Board vs simulator: same detection count, and greedy one-to-one matching
by IoU with the same class and IoU >= iou_min for every detection."""
import json
import sys

def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0

def compare(sim, board, iou_min=0.9):
    errs = []
    for f, sd in sim.items():
        if f not in board:
            errs.append(f"{f}: missing from board output")
            continue
        bd = list(board[f])
        if len(sd) != len(bd):
            errs.append(f"{f}: {len(sd)} sim dets vs {len(bd)} board dets")
            continue
        for s in sorted(sd, key=lambda d: -d[5]):
            best = max(bd, key=lambda b: iou(s, b) if b[4] == s[4] else -1.0, default=None)
            if best is None or best[4] != s[4] or iou(s, best) < iou_min:
                errs.append(f"{f}: no board match for class {s[4]} box {s[:4]}")
                continue
            bd.remove(best)
    return errs

if __name__ == "__main__":
    errs = compare(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
    print("\n".join(errs) or "board matches simulator")
    sys.exit(1 if errs else 0)
