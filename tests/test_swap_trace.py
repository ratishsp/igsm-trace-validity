"""swap_solutions: B's trace lands between A's problem and A's answer,
symmetrically, and token_id structure stays well-formed."""
import igsm_paths  # noqa: F401  (sys.path setup)
from data_gen.pretrain.id_gen import IdGen
from tools.tools import fix_seed
from traces.swap_trace import swap_solutions, TOKEN_START_PROBLEM, TOKEN_START_SOLUTION, \
    TOKEN_START_ANSWER, TOKEN_EOS


def gen():
    g = IdGen(max_op=15, max_edge=20, op=None, perm_level=5, detail_level=0)
    g.gen_prob(list(range(17)), p_format="pq")
    return g


def test_swap_trace():
    fix_seed(9)
    for trial in range(20):
        a, b = gen(), gen()
        pa, sa, ansa = list(a.prob_token), list(a.sol_token), list(a.ans_token)
        pb, sb, ansb = list(b.prob_token), list(b.sol_token), list(b.ans_token)
        swap_solutions(a, b)
        exp_a = [TOKEN_START_PROBLEM] + pa + [TOKEN_START_SOLUTION] + sb + \
                [TOKEN_START_ANSWER] + ansa + [TOKEN_EOS]
        exp_b = [TOKEN_START_PROBLEM] + pb + [TOKEN_START_SOLUTION] + sa + \
                [TOKEN_START_ANSWER] + ansb + [TOKEN_EOS]
        assert a.token_id == exp_a and b.token_id == exp_b, \
            f"trial {trial}: token_id not [own problem][other sol][own answer]"
        assert a.sol_token == sb and b.sol_token == sa, f"trial {trial}: sol_token not exchanged"

    # double swap restores the originals
    fix_seed(10)
    a, b = gen(), gen()
    orig_a, orig_b = list(a.token_id), list(b.token_id)
    swap_solutions(a, b); swap_solutions(a, b)
    assert a.token_id == orig_a and b.token_id == orig_b, "double swap is not the identity"


if __name__ == "__main__":
    test_swap_trace(); print("PASS")
