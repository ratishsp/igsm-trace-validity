"""Op-matched swap: every swapped pair must share one op count, and pairs
must span multiple op values across the stream."""
import igsm_paths
import igsm_train
from traces import swap_trace


def test_swap_match_op():
    cfg = dict(igsm_train.CONFIG, swap_traces=True, swap_match_op=True,
               seed=7, igsm_root=igsm_paths.IGSM_ROOT)
    cfg = igsm_train._apply_overrides(cfg)

    pairs = []
    orig = swap_trace.swap_solutions
    def spy(a, b):
        pairs.append((a.problem.n_op, b.problem.n_op))
        return orig(a, b)
    swap_trace.swap_solutions = spy
    try:
        ds = igsm_train.IGSMIterableDataset(cfg, rank=0)
        it = iter(ds)
        while len(pairs) < 30:
            next(it)
    finally:
        swap_trace.swap_solutions = orig

    assert all(x == y for x, y in pairs), pairs
    assert len({x for x, _ in pairs}) >= 3, f"suspiciously few op values: {pairs}"
    print(f"OK: {len(pairs)} pairs, all op-matched; ops seen: {sorted({x for x, _ in pairs})}")


if __name__ == "__main__":
    test_swap_match_op()
