"""Deterministic, disjoint COCO val2017 subsets: calib 500, eval 50, board 10."""
import argparse
import random
from pathlib import Path

SEED = 20261007

def select(names: list[str], seed: int = SEED) -> dict[str, list[str]]:
    pool = sorted(names)
    random.Random(seed).shuffle(pool)
    return {"calib": sorted(pool[:500]), "eval": sorted(pool[500:550]), "board": sorted(pool[550:560])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "lists"))
    a = ap.parse_args()
    s = select([p.name for p in Path(a.val_dir).glob("*.jpg")])
    Path(a.out).mkdir(exist_ok=True)
    for k, v in s.items():
        (Path(a.out) / f"{k}.txt").write_text("\n".join(v) + "\n")

if __name__ == "__main__":
    main()
