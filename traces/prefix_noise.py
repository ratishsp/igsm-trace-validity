"""Shuffled-prefix training traces: the tokens of the first frac of the solution
sentences (at least one sentence, at most n-1; frac 1.0 = the whole trace) are
randomly permuted. The rest of the trace, the problem and the answer are untouched.
"""
import random

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


def split_index(n_sentences, frac):
    if frac >= 1.0:
        return n_sentences            # whole trace destroyed, single-sentence traces included
    if n_sentences < 2:
        return 0
    return min(max(1, round(n_sentences * frac)), n_sentences - 1)


def noise_prefix(id_gen, frac, rng=None):
    """Returns the number of sentences destroyed (0 if the trace is untouched)."""
    from tools.tools import tokenizer
    rng = rng or random
    sents = list(id_gen.problem.solution)
    k = split_index(len(sents), frac)
    if k == 0:
        return 0
    enc = lambda ss: tokenizer.encode(" " + ". ".join(ss) + ".", return_tensors="pt")[0].tolist()
    block = enc(sents[:k])
    rest = enc(sents[k:]) if k < len(sents) else []
    perm = list(block)
    for _ in range(10):
        rng.shuffle(perm)
        if perm != block:
            break
    id_gen.sol_token = perm + rest
    _rebuild(id_gen)
    return k
