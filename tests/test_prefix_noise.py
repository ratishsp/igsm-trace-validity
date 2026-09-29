"""--prefix_noise_frac: the first k sentences' tokens are permuted; the rest of
the trace, the problem and the answer are untouched. Split rule: k covers the
first FRAC of the sentences, at least one, at most n-1; frac = 1.0 is the whole
trace."""
import random
import igsm_paths
import igsm_train
from traces.prefix_noise import noise_prefix, split_index
from tools.tools import tokenizer, fix_seed

base = igsm_train._apply_overrides(dict(igsm_train.CONFIG, seed=3,
                                         igsm_root=igsm_paths.IGSM_ROOT))


def test_split_index():
    assert [split_index(n, 0.3) for n in (1, 2, 4, 10)] == [0, 1, 1, 3]
    assert [split_index(n, 0.5) for n in (1, 2, 4, 10)] == [0, 1, 2, 5]
    assert split_index(3, 0.9) == 2          # below 1.0: never the whole trace
    assert [split_index(n, 1.0) for n in (1, 2, 10)] == [1, 2, 10]   # 1.0: the whole trace


def test_whole_trace():
    fix_seed(5); rng1 = random.Random(2)
    for _ in range(30):                       # frac = 1.0: bag of the right tokens, nothing valid left
        g = igsm_train._make_id_gen(base); g.gen_prob(list(range(17)), p_format="pq")
        gold, ans = list(g.sol_token), list(g.ans_token)
        k = noise_prefix(g, 1.0, rng1)
        assert k == len(g.problem.solution) and sorted(g.sol_token) == sorted(gold) and g.ans_token == ans
        assert g.sol_token != gold or len(set(gold)) == 1


def test_prefix():
    fix_seed(7); rng = random.Random(1)
    n_noised = n_single = 0
    for frac in (0.1, 0.3, 0.5):
        for _ in range(40):
            g = igsm_train._make_id_gen(base)
            g.gen_prob(list(range(17)), p_format="pq")
            gold_sol, gold_prob, gold_ans = list(g.sol_token), list(g.prob_token), list(g.ans_token)
            sents = list(g.problem.solution)
            k = noise_prefix(g, frac, rng)
            assert g.prob_token == gold_prob and g.ans_token == gold_ans
            if k == 0:
                assert len(sents) == 1 and g.sol_token == gold_sol; n_single += 1; continue
            n_noised += 1
            assert k == split_index(len(sents), frac)
            assert sorted(g.sol_token) == sorted(gold_sol), "token multiset must be preserved"
            rest = tokenizer.encode(" " + ". ".join(sents[k:]) + ".", return_tensors="pt")[0].tolist()
            assert g.sol_token[-len(rest):] == rest, "suffix must be the gold remainder"
            assert gold_sol[-len(rest):] == rest, "split tokenization must match the joint one"
            assert g.sol_token[:-len(rest)] != gold_sol[:-len(rest)], "prefix must differ from gold"
            assert g.token_id[-len(gold_ans) - 1:-1] == gold_ans
    print(f"ok: {n_noised} noised, {n_single} single-sentence traces left unchanged")


def test_dispatch():
    cfg = dict(base, prefix_noise_frac=0.3)
    assert igsm_train._is_corrupted(cfg) and "token-shuffled" in igsm_train._corruption_label(cfg)
    fix_seed(9); g = igsm_train._make_id_gen(cfg); g.gen_prob(list(range(17)), p_format="pq")
    while len(g.problem.solution) < 4:
        g = igsm_train._make_id_gen(cfg); g.gen_prob(list(range(17)), p_format="pq")
    gold = list(g.sol_token)
    assert igsm_train._maybe_corrupt(cfg, g) > 0 and g.sol_token != gold
    print("EXAMPLE:", tokenizer.decode(g.sol_token)[:400])


if __name__ == "__main__":
    test_split_index(); test_whole_trace(); test_prefix(); test_dispatch(); print("PASS")
