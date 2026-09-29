"""Causes table: why the trace is invalid when the answer is correct. Each such
row is assigned to the first matching cause:

  1. trace does not parse
  2. internal error (undefined reference, wrong arithmetic, wrong order)
  3. defines a name not in the problem
  4. a step uses the wrong inputs, split by how the answer survived:
       - the wrong input holds the same value as the missing one
       - the wrongly computed value is later multiplied by zero
       - the wrongly computed value reaches the answer and it still matches
       - no later step uses the wrongly computed value

Usage: python analysis/ic_causes.py <dir with op_*.json[.gz]> [--levels 15,20,21,22,23]
"""
import argparse, gzip, json, os, re, sys

SENT = re.compile(r"Define ((?:(?!Define )[^;])+?) as ([A-Za-z]); (.*?)\.(?= Define |$)")

def checker_valid(r):
    """The iGSM checker's full verdict: the trace parses, passes the reference,
    calculation and order checks, and its dependency sets match the problem's
    (num_non_appear == 0 and num_incorrect_elements == 0)."""
    if "num_incorrect_elements" not in r:
        raise KeyError("row has no num_incorrect_elements: this evaluation file was written without "
                       "the problem-dependent checker verdicts and cannot be scored for trace validity")
    internal = (bool(r.get("correct_parse")) and not r.get("incorrect_refer") and not r.get("incorrect_calcs")
                and not r.get("excess_definitions") and not r.get("incorrect_order"))
    return internal and r.get("num_non_appear", 0) == 0 and r.get("num_incorrect_elements", 0) == 0

def parse_steps(sol):
    """[(name, sym, [(clause_sym, operand_syms, result)])] with symbols resolved per clause."""
    steps = []
    for m in SENT.finditer(sol):
        name, sym, body = m.group(1), m.group(2), m.group(3)
        clauses = []
        for clause in body.split("; "):
            c = clause[3:] if clause.startswith("so ") else clause
            parts = c.split(" = ")
            if len(parts) < 2: return None
            try: result = int(parts[-1])
            except ValueError: return None
            mid = parts[1:-1]
            toks = [t for t in (mid[0].split(" ") if mid else []) if t not in "+-*"]
            clauses.append((parts[0], [t for t in toks if not t.isdigit()], result))
        steps.append((name, sym, clauses))
    return steps

def classify(r):
    st = r["problem_struct"]; gold = {q["name"]: q for q in st["params"]}
    if not r.get("correct_parse"): return "trace does not parse"
    if r.get("incorrect_refer") or r.get("incorrect_calcs") or r.get("incorrect_order") or r.get("excess_definitions"):
        return "internal error"
    steps = parse_steps(r["solution"])
    if steps is None: return "trace does not parse"
    sym2name = {}
    for name, sym, clauses in steps:
        if name not in gold: return "defines a name not in the problem"
        local = {}
        used = set()
        for csym, opsyms, result in clauses:
            for t in opsyms:
                if t in local: continue
                if t in sym2name: used.add(sym2name[t])
                else: return "internal error"
            local[csym] = result
        want = set(gold[name]["parents"])
        if used != want:
            # how did the answer survive the wrong inputs?
            extra = sorted(used - want); missing = sorted(want - used)
            if (len(extra) == 1 and len(missing) == 1 and extra[0] in gold and missing[0] in gold
                    and gold[extra[0]]["value"] == gold[missing[0]]["value"]):
                return "wrong inputs: same value"
            wrong_val = clauses[-1][2]; gold_val = gold[name]["value"]
            if wrong_val == gold_val:
                return "wrong inputs: same value"          # different inputs, identical result
            # follow the wrong value forward: a clause that reads a tainted symbol
            # produces a tainted result, unless it multiplies it by a value that is 0
            # (then the product is 0 whatever the tainted value was) or the tainted
            # symbol is added to nothing else and the result equals the gold value.
            tainted = {sym}
            killed_by_zero = False
            values = {}                       # symbol -> numeric value, all steps so far
            for n0, s0, cl0 in steps[:steps.index((name, sym, clauses)) + 1]:
                for csym, opsyms, res in cl0: values[csym] = res
            later = steps[steps.index((name, sym, clauses)) + 1:]
            for n2, s2, cl2 in later:
                for csym, opsyms, res in cl2:
                    hit = [t for t in opsyms if t in tainted]
                    if hit:
                        others = [t for t in opsyms if t not in tainted]
                        if res == 0 and any(values.get(t) == 0 for t in others) and len(opsyms) == 2 and "*" in _clause_ops(r, csym):
                            killed_by_zero = True          # tainted factor times zero
                        else:
                            tainted.add(csym)
                    values[csym] = res
                if s2 in tainted or cl2[-1][0] in tainted: tainted.add(s2)
            final_sym = steps[-1][1]
            if final_sym in tainted or steps[-1][2][-1][0] in tainted:
                return "wrong inputs: reaches the answer, still matches"
            if len(tainted) == 1 and not killed_by_zero:
                return "wrong inputs: never used again"
            return "wrong inputs: multiplied by zero"
        sym2name[sym] = name
    return "no wrong step found"

def _clause_ops(r, csym):
    """operators used in the clause that defines csym (looked up in the raw trace)."""
    m = re.search(r"(?:^|; |so )" + re.escape(csym) + r" = ([^=]*?) = ", r["solution"])
    return re.findall(r"[+\-*]", m.group(1)) if m else []

ROWS = ["a step uses the wrong inputs", "wrong inputs: same value", "wrong inputs: multiplied by zero",
        "wrong inputs: reaches the answer, still matches", "wrong inputs: never used again",
        "trace does not parse", "internal error", "defines a name not in the problem", "no wrong step found"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--levels", default="15,20,21,22,23")
    a = ap.parse_args(); levels = a.levels.split(",")
    table = {row: [] for row in ["correct answers", "with an invalid trace"] + ROWS}
    for lvl in levels:
        path = os.path.join(a.dir, f"op_{lvl}.json")
        if not os.path.exists(path) and os.path.exists(path + ".gz"):
            b = json.load(gzip.open(path + ".gz", "rt"))
        else:
            b = json.load(open(path))
        rows = b["results"] if isinstance(b, dict) else b
        cor = [r for r in rows if r["correct"]]; ic = [r for r in cor if not checker_valid(r)]
        counts = {row: 0 for row in ROWS}
        for r in ic:
            k = classify(r); counts[k] += 1
            if k.startswith("wrong inputs"): counts["a step uses the wrong inputs"] += 1
        table["correct answers"].append(len(cor)); table["with an invalid trace"].append(len(ic))
        for row in ROWS: table[row].append(counts[row])
    w = max(len(k) for k in table)
    print(f"{'':{w}s} " + " ".join(f"op{l:>3s}" for l in levels))
    for k, v in table.items(): print(f"{k:{w}s} " + " ".join(f"{x:5d}" for x in v))

if __name__ == "__main__":
    main()
