"""--no_trace (answer-only arm): the training sequence is exactly
[PROB] problem [SOL] [ANS] answer [EOS]; the dataset stream emits it; collate
supervises [ANS], the answer and the EOS only."""
import torch
import igsm_paths
import igsm_train
from igsm_train import (drop_trace, collate_fn, TOKEN_START_PROBLEM, TOKEN_START_SOLUTION,
                        TOKEN_START_ANSWER, TOKEN_EOS)
from tools.tools import fix_seed

base = igsm_train._apply_overrides(dict(igsm_train.CONFIG, seed=3,
                                         igsm_root=igsm_paths.IGSM_ROOT))


def test_drop_trace():
    fix_seed(21)
    for _ in range(10):
        g = igsm_train._make_id_gen(base); g.gen_prob(list(range(17)), p_format="pq")
        prob, ans = list(g.prob_token), list(g.ans_token)
        assert g.sol_token, "generator produced an empty trace"
        assert drop_trace(g) == 1
        assert g.sol_token == [] and g.sol == ""
        assert g.prob_token == prob and g.ans_token == ans
        assert g.token_id == [TOKEN_START_PROBLEM] + prob + [TOKEN_START_SOLUTION] \
            + [TOKEN_START_ANSWER] + ans + [TOKEN_EOS]


def test_dispatch_and_label():
    cfg = dict(base, no_trace=True)
    assert igsm_train._is_corrupted(cfg) and "no trace" in igsm_train._corruption_label(cfg)
    assert not igsm_train._is_corrupted(base) and igsm_train._corruption_label(base) == "clean"
    fix_seed(22)
    g = igsm_train._make_id_gen(cfg); g.gen_prob(list(range(17)), p_format="pq")
    assert igsm_train._maybe_corrupt(cfg, g) == 1 and g.sol_token == []
    h = igsm_train._make_id_gen(base); h.gen_prob(list(range(17)), p_format="pq")
    assert igsm_train._maybe_corrupt(base, h) == 0 and h.sol_token


def test_stream_and_collate():
    cfg = dict(base, no_trace=True)
    ds = igsm_train.IGSMIterableDataset(cfg, rank=0)
    it = iter(ds)
    buf, seen = [], 0
    while seen < 8:
        buf += next(it)["input_ids"].tolist()
        while True:
            try:
                s_p = buf.index(TOKEN_START_PROBLEM)
                s_s = buf.index(TOKEN_START_SOLUTION, s_p)
                s_a = buf.index(TOKEN_START_ANSWER, s_s)
                s_e = buf.index(TOKEN_EOS, s_a)
            except ValueError:
                break
            assert s_a == s_s + 1, "tokens between [SOL] and [ANS] in a no-trace example"
            assert 1 <= s_e - s_a - 1 <= 3, "answer span has an unexpected length"
            seen += 1
            buf = buf[s_e + 1:]
    # supervision: exactly [ANS], the answer tokens and EOS carry labels
    seq = torch.tensor([TOKEN_START_PROBLEM, 300, 301, 302, TOKEN_START_SOLUTION,
                        TOKEN_START_ANSWER, 1270, TOKEN_EOS])
    lab = collate_fn([{"input_ids": seq}])["labels"][0].tolist()
    assert lab == [-100, -100, -100, -100, -100, TOKEN_START_ANSWER, 1270, TOKEN_EOS]
    print(f"ok: {seen} answer-only examples in the stream, supervision on [ANS]+answer+EOS")


if __name__ == "__main__":
    test_drop_trace(); test_dispatch_and_label(); test_stream_and_collate(); print("PASS")
