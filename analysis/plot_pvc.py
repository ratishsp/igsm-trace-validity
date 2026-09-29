"""Validity against op for one model: answer accuracy, answer correct and trace
valid, and the share of correct answers whose trace is valid.

Usage: python analysis/plot_pvc.py <model_dir> [<out_prefix>]
"""
import glob, gzip, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ic_causes import checker_valid

d = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "pvc_vs_op"
def load(path):
    b = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    return b["results"] if isinstance(b, dict) else b

per_op = {}
for f in sorted(glob.glob(os.path.join(d, "in_dist_mixture", "*.json*"))):
    for r in load(f):
        per_op.setdefault(r["n_op"], []).append(r)
for lvl in (15, 20, 21, 22, 23):
    per_op[lvl] = load(glob.glob(os.path.join(d, f"op_{lvl}.json*"))[0])

rows = []
for op in sorted(per_op):
    rs = per_op[op]
    if op < 10 or op == 14 or len(rs) < 50: continue
    cor = sum(r["correct"] for r in rs); vc = sum(r["correct"] and checker_valid(r) for r in rs)
    rows.append((op, len(rs), 100 * cor / len(rs), 100 * vc / len(rs), 100 * vc / max(1, cor)))
    print(f"op {op:2d} n={len(rs):4d} acc={rows[-1][2]:5.1f} joint={rows[-1][3]:5.1f} P(valid|correct)={rows[-1][4]:5.1f}")

ops = [r[0] for r in rows]; xpos = []; x = 0
for i, op in enumerate(ops):
    if i > 0 and op - ops[i - 1] > 1: x += 1.0
    xpos.append(x); x += 1
acc = [r[2] for r in rows]; joint = [r[3] for r in rows]; pca = [r[4] for r in rows]
fig, ax = plt.subplots(figsize=(6.4, 2.3), dpi=200)
blue, red, grey = "#1f5fa8", "#c8401e", "#6b6b6b"; i15 = ops.index(15)
ax.axvspan(xpos[i15] + 0.9, xpos[-1] + 0.5, color="#000000", alpha=0.05, lw=0)
ax.plot(xpos, acc, "-o", color=blue, lw=2, ms=5, label="answer correct")
ax.plot(xpos, pca, "-s", color=red, lw=2, ms=5, label="trace valid, given answer correct")
ax.plot(xpos, joint, "--^", color=grey, lw=1.8, ms=5.5, label="answer correct and trace valid")
for y, c in ((acc[-1], blue), (pca[-1], red), (joint[-1], grey)):
    ax.text(xpos[-1] + 0.25, y, f"{y:.1f}", color=c, fontsize=11, va="center")
ax.set_xticks(xpos); ax.set_xticklabels([str(o) for o in ops], fontsize=11)
ax.set_xlim(-0.5, xpos[-1] + 1.3); ax.set_ylim(34, 106); ax.set_yticks([40, 60, 80, 100]); ax.tick_params(axis="y", labelsize=11)
ax.set_xlabel("operations in the necessary computation (op)", fontsize=12); ax.set_ylabel("percent", fontsize=12)
ax.text(xpos[i15] / 2, 103, "in distribution", ha="center", va="bottom", fontsize=11, color="#444")
ax.text((xpos[i15] + 1 + xpos[-1]) / 2, 103, "out of distribution", ha="center", va="bottom", fontsize=11, color="#444")
ax.grid(axis="y", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(loc="lower left", fontsize=10.5, frameon=False); fig.tight_layout(pad=0.3)
fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
