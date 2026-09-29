"""Answer accuracy per model and level: accuracy.csv and in_dist_per_op.csv.

Usage: python analysis/summarize.py <root> [<out_dir>]
  <root>/<model>/op_*.json[.gz]                  exact-op levels (evaluate.py --only_op)
  <root>/<model>/in_dist_mixture/*.json[.gz]     mixture chunks (evaluate.py --only_level 'op<=15')
"""
import csv, glob, gzip, json, os, re, sys
from collections import defaultdict

root = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "."

def load(p):
    b = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
    return b["results"] if isinstance(b, dict) else b

def steps(sol):
    return len([s for s in re.split(r"[;.]", sol) if "=" in s])  # op units

acc_rows, perop_rows = [], []
for mdir in sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)):
    model = os.path.basename(mdir)
    # exact-op levels
    for f in sorted(glob.glob(f"{mdir}/op_*.json*")):
        lev = os.path.basename(f).split(".json")[0]
        rows = load(f)
        n = len(rows); c = sum(bool(r["correct"]) for r in rows)
        med = sorted(steps(r["solution"]) for r in rows)[n // 2]
        acc_rows.append([model, lev, n, f"{100*c/n:.1f}", med])
    # in-distribution mixture
    chunks = sorted(glob.glob(f"{mdir}/in_dist_mixture/*.json*"))
    if chunks:
        rows = sum((load(f) for f in chunks), [])
        n = len(rows); c = sum(bool(r["correct"]) for r in rows)
        deep = [r for r in rows if r.get("n_op", 0) >= 8]
        dc = sum(bool(r["correct"]) for r in deep)
        med = sorted(steps(r["solution"]) for r in rows)[n // 2]
        acc_rows.append([model, "op_le_15_mixture", n, f"{100*c/n:.1f}", med])
        acc_rows.append([model, "op_ge_8_slice", len(deep), f"{100*dc/len(deep):.1f}", ""])
        byop = defaultdict(list)
        for r in rows:
            byop[r.get("n_op", 0)].append(bool(r["correct"]))
        for op in sorted(byop):
            perop_rows.append([model, op, len(byop[op]), f"{100*sum(byop[op])/len(byop[op]):.1f}"])

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "accuracy.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["model", "level", "n", "accuracy", "median_steps"]); w.writerows(acc_rows)
with open(os.path.join(out_dir, "in_dist_per_op.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["model", "op", "n", "accuracy"]); w.writerows(perop_rows)
print(open(os.path.join(out_dir, "accuracy.csv")).read())
