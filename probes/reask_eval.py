"""Re-ask (Ye et al.): keep the problem text and re-point the question at another
parameter of the same world. Wraps tools_test.re_ask; only parameters whose
stated ancestors are in the text are askable, and the new query's op is
recounted on the whole template. Returns None if no candidate works.
"""
import copy
import networkx as nx


def askable_params(p):
    stated = set(x for x in p.problem_order if x[0] == 0)
    out = []
    for x in p.all_param:
        if x == p.ques_idx or (x[0] == 0 and x not in stated):
            continue
        anc = nx.ancestors(p.whole_template, x)
        if all(a[0] != 0 or a in stated for a in anc if a != p.rand):
            out.append(x)
    return out


def n_op_whole(p, order):
    n = 0
    for x in order:
        pre = [y for y in p.whole_template.predecessors(x) if y != p.rand]
        n += 1 if len(pre) <= 2 else len(pre) - 1
    return n


def reask(id_gen, hash_val, rng):
    """Re-point id_gen's problem at a random askable parameter, in place.
    Returns a dict describing the change, or None if no target worked."""
    from tools.tools_test import re_ask
    p = id_gen.problem
    orig = dict(orig_query=p.ques_idx, orig_chain=list(p.topological_order), orig_op=int(p.n_op))
    cands = askable_params(p)
    rng.shuffle(cands)
    for target in cands[:5]:
        q = copy.deepcopy(p)
        try:
            re_ask(q, param=target)
        except Exception:
            pass   # re_ask's final op recount can raise (see module docstring);
                   # the solution is already built, and n_op is recounted below
        if not q.solution or q.ques_idx != target:
            continue
        q.n_op = n_op_whole(q, q.topological_order)
        id_gen.gen_prob(hash_val, p_format="pq", problem=q)
        return dict(orig, new_query=target, new_op=int(q.n_op),
                    downstream_of_query=set(nx.descendants(q.whole_template, target)))
    return None
