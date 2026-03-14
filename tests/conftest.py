"""
conftest.py
-----------
Shared pytest fixtures for mechanicsdsl-datasets test suite.
"""
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def repo_root():
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def datasets_dir(repo_root):
    return repo_root / "datasets"


@pytest.fixture(scope="session")
def all_dataset_dirs(datasets_dir):
    return sorted([d for d in datasets_dir.iterdir() if d.is_dir()])
