"""Re-ask analysis from evaluate.py --reask outputs: per cell and new-op bucket,
accuracy, share of correct traces that define an unnecessary parameter, unnecessary
parameters per correct trace, and where those parameters sit relative to the new query.

  --where        inclusion of non-necessary parameters by kind, per cell
  --stated_only  the same, restricted to stated parameters
  --examples     example traces per cell

Usage: python analysis/reask_analysis.py --evals_dir <dir with reask_768/ and geom_sweep_768/> [--where | --stated_only | --examples]
"""
import argparse, glob, gzip, json, os, re
from collections import Counter

DEF = re.compile(r"Define (.+?) as [A-Za-z]")
BUCKETS_REASK = [(1, 3), (4, 7), (8, 12), (13, 99)]
KINDS = ("original chain, not downstream", "downstream of new query", "unrelated distractor")


def load_cell(evals_dir, subdir, cell):
    paths = sorted(glob.glob(os.path.join(evals_dir, subdir, cell, "op_*.json*")))
    if not paths:
        raise SystemExit(f"no op_*.json under {os.path.join(evals_dir, subdir, cell)}")
    f = gzip.open(paths[0], "rt") if paths[0].endswith(".gz") else open(paths[0])
    return json.load(f)["results"]


def kind(p):
    return ("downstream of new query" if p.get("downstream_of_query") else
            "original chain, not downstream" if p.get("in_original_chain") else "unrelated distractor")


def kind_long(p):
    return ("downstream of the new query" if p.get("downstream_of_query") else
            "on the ORIGINAL chain (not an ancestor or descendant of the new query)"
            if p.get("in_original_chain") else "unrelated distractor")


def struct(r):
    return {p["name"]: p for p in r["problem_struct"]["params"]}


def extras(r, st=None):
    """Names the trace defines that are parameters of the problem but not necessary."""
    st = st or struct(r)
    return [d.strip() for d in DEF.findall(r["solution"]) if d.strip() in st and not st[d.strip()]["necessary"]]


# -- default: per new-op bucket table ---------------------------------------

def analyse(rows, label, reask=True):
    buckets = BUCKETS_REASK if reask else [(0, 99)]
    print(f"== {label}: n={len(rows)} acc={100*sum(r['correct'] for r in rows)/len(rows):.1f}")
    where = Counter()
    for lo, hi in buckets:
        sub = [r for r in rows if lo <= r["n_op"] <= hi]
        if len(sub) < 20:
            continue
        cor = [r for r in sub if r["correct"]]
        extra_per_correct = []
        padded = 0
        for r in cor:
            st = struct(r)
            extra = extras(r, st)
            extra_per_correct.append(len(extra)); padded += bool(extra)
            for d in extra:
                p = st[d]
                where["downstream of the new query" if p.get("downstream_of_query") else
                      "on original chain, not downstream" if p.get("in_original_chain") else "unrelated distractor"] += 1
        slack = sorted(r["n_stated"] - r["n_needed_stated"] for r in sub)[len(sub) // 2]
        print(f"   new op {lo:2d}-{hi:<2d} n={len(sub):4d} acc={100*len(cor)/len(sub):5.1f} | "
              f"median irrelevant sentences {slack:2d} | padded correct traces {100*padded/max(1,len(cor)):5.2f}% | "
              f"unnecessary params per correct solution {sum(extra_per_correct)/max(1,len(cor)):.3f} | "
              f"wrong-parent {100*sum(r['num_incorrect_elements']>0 for r in sub)/len(sub):4.1f}%")
    if where:
        print("   where the unnecessary params sit:", dict(where))


# -- --where / --stated_only: inclusion by kind -----------------------------

def inclusion(rows, cell, stated_only):
    rows = [r for r in rows if r["correct"]]
    avail = Counter(); incl = Counter()
    for r in rows:
        st = struct(r)
        defined = set(d.strip() for d in DEF.findall(r["solution"]))
        for n, p in st.items():
            if p["necessary"] or (stated_only and not p["stated"]):
                continue
            avail[kind(p)] += 1
            if n in defined:
                incl[kind(p)] += 1
    if stated_only:
        print(f"== {cell}, STATED parameters only")
        for k in KINDS:
            a, i = avail[k], incl[k]
            print(f"   {k:32s} available {a:6d} | included {i:4d} | inclusion rate {100*i/max(1,a):5.2f}%")
    else:
        print(f"== {cell} (correct solutions: {len(rows)})")
        for k in KINDS:
            a, i = avail[k], incl[k]
            print(f"   {k:32s} available {a:6d} ({100*a/sum(avail.values()):4.1f}% of non-necessary) | "
                  f"included {i:4d} ({100*i/max(1,sum(incl.values())):4.1f}% of included) | "
                  f"inclusion rate {100*i/max(1,a):5.2f}%")


def padded_example(rows, cell):
    """One example: re-ask, shallow new op, heavily padded."""
    for r in rows:
        st = struct(r)
        extra = extras(r, st)
        if r["correct"] and r["n_op"] <= 3 and len(extra) >= 4:
            print(f"\nEXAMPLE ({cell}, reask):", r["reask"])
            print("necessary:", [n for n, p in st.items() if p["necessary"]])
            print("unnecessary defined:", [(d, kind(st[d])) for d in extra])
            print("TRACE:", r["solution"][:900])
            return


# -- --examples: full traces --------------------------------------------------

def show(r, title):
    st = struct(r)
    extra = extras(r, st)
    rk = r["reask"]
    print("-" * 110 + f"\n{title}")
    print(f"original question (op {rk['orig_op']}): {rk['orig_query']}   ->   new question (op {rk['new_op']}): {rk['new_query']}")
    print(f"sentences in problem {r['n_stated']} | needed for the new question {r['n_needed']} | "
          f"gold {r['ground_truth']} model {r['predicted']} correct={r['correct']}")
    print("\nPROBLEM:\n" + r["problem"])
    print("\nNECESSARY FOR THE NEW QUESTION:")
    for p in r["problem_struct"]["params"]:
        if p["necessary"]:
            print(f"  {p['name']}  <-  {p['parents'] if p['parents'] else '(given)'}  = {p['value']}")
    print("\nMODEL TRACE:\n" + r["solution"])
    print(f"\nUNNECESSARY PARAMETERS THE MODEL DEFINED ({len(extra)}):")
    for d in extra:
        print(f"  {d}   [{kind_long(st[d])}]")
    print()


def cell_label(cell, model_label):
    if cell.startswith("std_"):
        return f"REASK, STANDARD WORLD ({model_label})"
    if cell.startswith("d4w44c600_"):
        return f"REASK, WIDE WORLD ({model_label}; widths 4, depth 4, 600-token cap)"
    return f"REASK, {cell} ({model_label})"


def examples(rows, label):
    cor = [r for r in rows if r["correct"]]
    pad = sorted([r for r in cor if len(extras(r)) > 0], key=lambda r: len(extras(r)))
    print("=" * 110 + f"\n{label}: {len(cor)} correct solutions, {len(pad)} padded\n" + "=" * 110)
    if not pad:
        print("(no padded correct traces in this cell)\n"); return
    show(pad[len(pad) // 2], "PADDED, median amount of padding")
    show(pad[-1], "PADDED, most padding in the cell")
    unpadded = next((r for r in cor if len(extras(r)) == 0 and 4 <= r["n_op"] <= 8), None)
    if unpadded is not None:
        show(unpadded, "NOT PADDED, for contrast")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals_dir", required=True, help="directory holding <reask_subdir>/ and <no_reask_subdir>/")
    ap.add_argument("--cells", nargs="+", default=["std_op15", "std_op20", "d4w44c600_op15", "d4w44c600_op20"],
                    help="re-ask cells")
    ap.add_argument("--no_reask_cells", nargs="+", default=["std_op10", "d4w44c600_op10", "std_op20", "d4w44c600_op20"],
                    help="plain exact-op cells shown for comparison (default table only)")
    ap.add_argument("--reask_subdir", default="reask_768")
    ap.add_argument("--no_reask_subdir", default="geom_sweep_768")
    ap.add_argument("--label", default="run A", help="model label used in the headings")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--where", action="store_true", help="inclusion of non-necessary parameters by kind")
    mode.add_argument("--stated_only", action="store_true", help="inclusion table over stated parameters only, plus one example")
    mode.add_argument("--examples", action="store_true", help="example traces per cell")
    a = ap.parse_args()

    reask_rows = {c: load_cell(a.evals_dir, a.reask_subdir, c) for c in a.cells}
    if a.where or a.stated_only:
        for c in a.cells:
            inclusion(reask_rows[c], c, stated_only=a.stated_only)
        if a.stated_only:
            padded_example(reask_rows[a.cells[-1]], a.cells[-1])
    elif a.examples:
        for c in a.cells:
            examples(reask_rows[c], cell_label(c, a.label))
    else:
        for c in a.cells:
            analyse(reask_rows[c], f"REASK {a.label}, {c}")
        print()
        for c in a.no_reask_cells:
            analyse(load_cell(a.evals_dir, a.no_reask_subdir, c), f"NO REASK {a.label}, {c}", reask=False)


if __name__ == "__main__":
    main()
