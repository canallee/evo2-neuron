import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run tests that load the real checkpoint or need a device")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    skip = pytest.mark.skip(reason="needs --runslow (real checkpoint or Neuron device)")

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
