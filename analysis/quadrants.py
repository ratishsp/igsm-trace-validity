"""Trace validity x answer correctness per level, and P(valid | correct).

Usage: python analysis/quadrants.py <dir with op_*.json[.gz]> [--levels 15,20,21,22,23]
"""
import argparse, gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ic_causes import checker_valid

def load_rows(d, lvl):
    path = os.path.join(d, f"op_{lvl}.json")
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        b = json.load(gzip.open(path + ".gz", "rt"))
    else:
        b = json.load(open(path))
    return b["results"] if isinstance(b, dict) else b

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--levels", default="15,20,21,22,23")
    a = ap.parse_args(); levels = a.levels.split(",")
    cells = {}
    for lvl in levels:
        rows = load_rows(a.dir, lvl)
        vc = sum(checker_valid(r) and r["correct"] for r in rows); vw = sum(checker_valid(r) and not r["correct"] for r in rows)
        ic = sum((not checker_valid(r)) and r["correct"] for r in rows); iw = sum((not checker_valid(r)) and not r["correct"] for r in rows)
        cells[lvl] = (vc, vw, ic, iw)
        print(f"op {lvl:>2s} n={len(rows)} valid&correct {vc:5d} valid&wrong {vw:4d} invalid&correct {ic:4d} invalid&wrong {iw:4d} | acc {100*(vc+ic)/len(rows):5.1f} | P(valid|correct) {100*vc/max(1,vc+ic):5.1f}")

if __name__ == "__main__":
    main()
