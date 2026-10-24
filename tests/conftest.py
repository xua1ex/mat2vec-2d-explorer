"""Tests marked `artifacts` need the locally built artifacts/ (pretrained Mat2vec model + build step).

Without them (e.g. a fresh clone) they are skipped with a reason; every other test still runs.
"""
from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FILES = ("vectors.npz", "candidates.parquet", "coords.parquet", "meta.json")


def pytest_configure(config):
    config.addinivalue_line("markers", "artifacts: needs artifacts/ built from the pretrained Mat2vec model")


def pytest_collection_modifyitems(config, items):
    if all((ARTIFACTS / f).exists() for f in FILES):
        return
    skip = pytest.mark.skip(reason="needs artifacts/: install the pretrained Mat2vec model and run "
                                   "python -m explorer.build_data (see README, full mode)")
    for item in items:
        if "artifacts" in item.keywords:
            item.add_marker(skip)
