"""Cross-problem trace swapping: problem A is trained with problem B's solution
trace and A's own answer, and vice versa.

    [PROB] problem_A [SOL] solution_B [ANS] answer_A [EOS]

With --swap_match_op the partner problem has the same op count.
"""

TOKEN_START_PROBLEM = 222
TOKEN_START_SOLUTION = 223
TOKEN_START_ANSWER = 224
TOKEN_EOS = 50256


def _rebuild(id_gen):
    id_gen.token_id = (
        [TOKEN_START_PROBLEM] + id_gen.prob_token +
        [TOKEN_START_SOLUTION] + id_gen.sol_token +
        [TOKEN_START_ANSWER] + id_gen.ans_token +
        [TOKEN_EOS]
    )


def swap_solutions(id_a, id_b):
    """Exchange the two problems' solution traces in place.

    Problems and answers stay with their own problem; only sol / sol_token
    cross over. Both token_id sequences are rebuilt.
    """
    id_a.sol, id_b.sol = id_b.sol, id_a.sol
    id_a.sol_token, id_b.sol_token = id_b.sol_token, id_a.sol_token
    _rebuild(id_a)
    _rebuild(id_b)
