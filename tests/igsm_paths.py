"""Locate the repository and the iGSM generator for the tests.

The generator is taken from IGSM_ROOT (env var, or `pytest --igsm_root PATH`),
else ./iGSM inside the repository. Both
directories are put on sys.path so `import igsm_train` and `from tools.tools
import tokenizer` work from any test.
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_igsm():
    cands = [os.environ.get("IGSM_ROOT"), os.path.join(REPO, "iGSM")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "data_gen")):
            return os.path.abspath(c)
    raise RuntimeError("iGSM generator not found: set IGSM_ROOT or run pytest --igsm_root PATH")


IGSM_ROOT = _find_igsm()
for d in (IGSM_ROOT, REPO):
    if d not in sys.path:
        sys.path.insert(0, d)
