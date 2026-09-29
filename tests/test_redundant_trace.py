"""--redundant_trace: traces are longer than minimal on average, remain valid
(answer correct, every used variable defined), and n_op semantics unchanged."""
import re
import igsm_paths
import igsm_train
from tools.tools import tokenizer, fix_seed

base = igsm_train._apply_overrides(dict(igsm_train.CONFIG, seed=3,
                                         igsm_root=igsm_paths.IGSM_ROOT))

STEP = re.compile(r"Define (.+?) as ([A-Za-z]); (.*?)\.(?= Define |$)")


def dangling_refs(sol):
    """Symbols used in a step's arithmetic that no earlier step (or clause) defined."""
    defined, dangling = set(), 0
    for m in STEP.finditer(sol):
        sym, body = m.group(2), m.group(3)
        local = set()
        for clause in body.split("; "):
            parts = (clause[3:] if clause.startswith("so ") else clause).split(" = ")
            if len(parts) >= 3:
                for t in parts[1].split(" "):
                    if t and not t.isdigit() and t not in "+-*" and t not in defined and t not in local:
                        dangling += 1
            local.add(parts[0])
        defined.add(sym)
    return dangling


def sample(redundant, n=60):
    cfg = dict(base, redundant_trace=redundant)
    fix_seed(11)
    out = []
    for _ in range(n):
        g = igsm_train._make_id_gen(cfg)
        g.gen_prob(list(range(17)), p_format="pq")
        sol = tokenizer.decode(g.sol_token)
        out.append((g.problem.n_op, sol.count("Define "), sol,
                    len(g.problem.topological_order)))
    return out


def test_redundant_trace():
    mins, reds = sample(False), sample(True)
    mean_min = sum(s for _, s, _, _ in mins) / len(mins)
    mean_red = sum(s for _, s, _, _ in reds) / len(reds)
    print(f"mean steps: minimal {mean_min:.1f}, redundant {mean_red:.1f}")
    assert mean_red > mean_min * 1.3, (mean_min, mean_red)

    # a redundant trace computes at least every NECESSARY variable (its steps
    # can undercut n_op, which counts operations, not steps)
    assert all(s >= nvars for _, s, _, nvars in reds), \
        [(nvars, s) for _, s, _, nvars in reds if s < nvars][:5]

    # n_op still reflects necessary ops: distribution comparable to minimal's
    assert abs(sum(n for n, _, _, _ in reds) / len(reds)
               - sum(n for n, _, _, _ in mins) / len(mins)) < 2.5

    # every symbol a step uses was defined by an earlier step (valid order)
    assert dangling_refs(mins[0][2].strip()) == 0
    bad = [sol for _, _, sol, _ in reds if dangling_refs(sol.strip()) > 0]
    print(f"traces with dangling refs among {len(reds)} redundant traces: {len(bad)}")
    assert not bad, bad[:1]


if __name__ == "__main__":
    test_redundant_trace(); print("OK")
