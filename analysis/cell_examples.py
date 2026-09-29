"""Example traces per validity x correctness cell from an evaluation file
(standard geometry, no re-ask).

Usage: python analysis/cell_examples.py <eval.json[.gz]> --op N [--per_cell 4] [--scan 600] [--igsm_root PATH]
"""
import argparse, gzip, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minimality import checker_valid
from evaluate import TEST_HASH_BIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("--op", type=int, default=None)
    ap.add_argument("--max_op", type=int, default=None); ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--max_edge", type=int, default=20); ap.add_argument("--per_cell", type=int, default=4)
    ap.add_argument("--scan", type=int, default=600)
    ap.add_argument("--igsm_root", default="./iGSM")
    a = ap.parse_args()
    sys.path.insert(0, a.igsm_root)
    from data_gen.pretrain.id_gen import IdGen
    from tools.tools import tokenizer, fix_seed
    from tools.sol_parser import Parser
    b = json.load(gzip.open(a.inp, "rt") if a.inp.endswith(".gz") else open(a.inp))
    rows = b["results"] if isinstance(b, dict) else b
    fix_seed(a.seed)
    cells = {"valid&correct": [], "valid&wrong": [], "invalid&correct": [], "invalid&wrong": []}
    for i, r in enumerate(rows[:a.scan]):
        g = IdGen(max_op=a.max_op or a.op, max_edge=a.max_edge, op=a.op, perm_level=5, detail_level=0)
        g.gen_prob([TEST_HASH_BIN[i % len(TEST_HASH_BIN)]], p_format="pq")
        if tokenizer.decode(g.prob_token).strip() != r["problem"].strip():
            continue
        cell = ("valid" if checker_valid(r) else "invalid") + "&" + ("correct" if r["correct"] else "wrong")
        if len(cells[cell]) >= a.per_cell:
            if all(len(v) >= a.per_cell for v in cells.values()):
                break
            continue
        flagged = {}
        try:
            p = Parser(r["solution"]); p.correct_refer(); p.correct_cal(); p.correct_order(); p.parse(g.problem)
            flagged = {"non_appear": p.non_appear_lst, "unnecessary": p.non_nece_lst, "wrong_parents": p.incorrect_lst}
        except Exception as e:
            flagged = {"parser_error": str(e)[:80]}
        cells[cell].append((i, r, flagged, " ".join(g.problem.solution)))
    for cell, items in cells.items():
        print(f"\n{'='*100}\nCELL {cell}: {len(items)} examples (of first {a.scan} rows)\n{'='*100}")
        for i, r, flagged, gold in items:
            print(f"\n--- row {i} | n_op {r.get('n_op')} | stated {r.get('n_stated')} needed {r.get('n_needed')} | gt {r['ground_truth']} pred {r['predicted']}")
            print("VERDICTS:", {k: r.get(k) for k in ["correct_parse", "incorrect_refer", "incorrect_calcs", "excess_definitions", "incorrect_order", "num_non_appear", "num_non_nece", "num_incorrect_elements"]})
            print("FLAGGED :", flagged)
            print("PROBLEM :", r["problem"])
            print("TRACE   :", r["solution"])
            print("GOLD    :", gold)


if __name__ == "__main__":
    main()
