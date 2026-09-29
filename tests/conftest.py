import os


def pytest_addoption(parser):
    parser.addoption("--igsm_root", default=None,
                     help="path to the iGSM generator checkout (else IGSM_ROOT, then ./iGSM)")


def pytest_configure(config):
    root = config.getoption("--igsm_root")
    if root:
        os.environ["IGSM_ROOT"] = os.path.abspath(root)
