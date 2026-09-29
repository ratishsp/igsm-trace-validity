"""--redundant_frac: the per-problem style choice hits both branches at the
requested rate."""
import igsm_paths
import igsm_train


def test_redundant_frac():
    calls = []
    orig = igsm_train._make_id_gen
    def spy(cfg, redundant=None):
        calls.append(redundant)
        return orig(cfg, redundant=redundant)
    igsm_train._make_id_gen = spy
    try:
        cfg = dict(igsm_train.CONFIG, seed=5, redundant_frac=0.9,
                   igsm_root=igsm_paths.IGSM_ROOT)
        cfg = igsm_train._apply_overrides(cfg)
        ds = igsm_train.IGSMIterableDataset(cfg, rank=0)
        it = iter(ds)
        while len([c for c in calls if c is not None]) < 100:
            next(it)
    finally:
        igsm_train._make_id_gen = orig
    flags = [c for c in calls if c is not None][:100]
    share = sum(flags) / len(flags)
    print(f"redundant share over {len(flags)} problems: {share:.2f}")
    assert 0.78 <= share <= 0.98, share
    assert any(f is False for f in flags), "minimal branch never taken"


if __name__ == "__main__":
    test_redundant_frac(); print("OK")
