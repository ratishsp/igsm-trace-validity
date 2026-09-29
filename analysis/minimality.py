"""Validity and minimality of emitted traces per level, from evaluation files.
valid = the iGSM checker's verdicts all pass; minimal = valid and no unnecessary
parameter defined; excess = defined minus necessary parameters over valid traces.

Usage: python analysis/minimality.py <dir with op_*.json[.gz]> [--by-slack]
"""
import glob, gzip, json, re, sys
from collections import defaultdict

DEF = re.compile(r"Define (.+?) as [A-Za-z]")
BUCKETS = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 99, "6+")]


def checker_valid(r):
    return (bool(r.get("correct_parse")) and not r.get("incorrect_refer")
            and not r.get("incorrect_calcs") and not r.get("excess_definitions")
            and not r.get("incorrect_order") and r.get("num_non_appear") == 0
            and r.get("num_incorrect_elements") == 0)


def summarize(rows):
    n = len(rows)
    c = [bool(r["correct"]) for r in rows]
    v = [checker_valid(r) for r in rows]
    m = [vi and r.get("num_non_nece") == 0 for vi, r in zip(v, rows)]
    ex = [len(DEF.findall(r["solution"])) - r["n_needed"] for vi, r in zip(v, rows) if vi]
    vc = sum(a and b for a, b in zip(v, c)); vw = sum(a and not b for a, b in zip(v, c))
    ic = sum((not a) and b for a, b in zip(v, c)); iw = n - vc - vw - ic
    return (f"n={n:5d} ans_acc={100*sum(c)/n:5.1f} checker_valid={100*sum(v)/n:5.1f} "
            f"minimal={100*sum(m)/n:5.1f} excess={sum(ex)/max(1,len(ex)):+.2f} "
            f"P(v|c)={100*vc/max(1,vc+ic):5.1f} [v&c={vc} v&w={vw} i&c={ic} i&w={iw}]")


def main():
    d = sys.argv[1]; by_slack = "--by-slack" in sys.argv
    files = [(f.split("/")[-1].split(".json")[0], [f]) for f in sorted(glob.glob(f"{d}/op_*.json*"))]
    mix = sorted(glob.glob(f"{d}/in_dist_mixture/seed*.json*"))
    if mix:
        files.insert(0, ("mixture", mix))
    for name, fs in files:
        rows = []
        for f in fs:
            b = json.load(gzip.open(f, "rt") if f.endswith(".gz") else open(f))
            rows += [r for r in (b["results"] if isinstance(b, dict) else b) if "n_needed" in r]
        if not rows:
            print(f"{name}: no rows with per-problem fields (n_needed)"); continue
        if name == "mixture":
            byop = defaultdict(list)
            for r in rows: byop[r["n_op"]].append(r)
            for op in sorted(byop):
                if len(byop[op]) >= 60: print(f"  op={op:2d}   {summarize(byop[op])}")
        else:
            print(f"  {name:6s}  {summarize(rows)}")
        if by_slack:
            for lo, hi, lab in BUCKETS:
                sub = [r for r in rows if lo <= r["n_stated"] - r["n_needed_stated"] <= hi]
                if len(sub) >= 30: print(f"      slack {lab:4s} {summarize(sub)}")


if __name__ == "__main__":
    main()
