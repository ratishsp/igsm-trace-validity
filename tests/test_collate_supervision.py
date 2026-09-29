"""collate_fn: labels reproduce the reference masking algorithm exactly
(loss on the [SOL] and [ANS] spans only, padding -100)."""
import random
import torch
import igsm_paths  # noqa: F401
from igsm_train import collate_fn, TOKEN_START_PROBLEM, TOKEN_START_SOLUTION, \
    TOKEN_START_ANSWER, TOKEN_EOS


def reference_masked(inp):
    """Reference implementation of the masking rule."""
    labels = torch.full_like(inp, -100)
    mask = torch.zeros(inp.size(0), dtype=torch.bool)
    supervising = False
    for j, tok in enumerate(inp.tolist()):
        if supervising:
            mask[j] = True
        if tok == TOKEN_START_SOLUTION or tok == TOKEN_START_ANSWER:
            supervising = True
        elif tok == TOKEN_START_PROBLEM or tok == TOKEN_EOS:
            supervising = False
    labels[mask] = inp[mask]
    return labels


def fake_window(rng, n_problems=3, total=120):
    toks = []
    for _ in range(n_problems):
        toks += [TOKEN_START_PROBLEM] + [rng.randrange(300, 40000) for _ in range(rng.randrange(8, 20))]
        toks += [TOKEN_START_SOLUTION] + [rng.randrange(300, 40000) for _ in range(rng.randrange(6, 15))]
        toks += [TOKEN_START_ANSWER, rng.randrange(300, 40000), TOKEN_EOS]
    return torch.tensor(toks[:total], dtype=torch.long)


def test_collate_supervision():
    rng = random.Random(7)
    batch = [{"input_ids": fake_window(rng)} for _ in range(6)]
    batch.append({"input_ids": fake_window(rng)[:57]})   # unequal lengths to exercise padding

    out = collate_fn(batch)
    for i, item in enumerate(batch):
        inp = item["input_ids"]; L = inp.size(0)
        assert torch.equal(out["labels"][i, :L], reference_masked(inp)), f"row {i}: differs from reference"
        assert (out["labels"][i, L:] == -100).all(), f"row {i}: padding labelled"
        assert torch.equal(out["input_ids"][i, :L], inp) and (out["input_ids"][i, L:] == TOKEN_EOS).all()
    n_sup = (out["labels"] != -100).sum().item(); n_tok = sum(b["input_ids"].numel() for b in batch)
    print(f"supervised-token share on synthetic windows: {n_sup / n_tok:.2f}")


if __name__ == "__main__":
    test_collate_supervision(); print("PASS")
