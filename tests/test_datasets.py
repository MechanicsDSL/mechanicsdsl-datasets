"""
test_datasets.py
----------------
Validates structure, schema, and data integrity for all datasets.

Run with:
    pytest tests/test_datasets.py -v
"""
import json
import h5py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASETS = sorted([d for d in (ROOT / "datasets").iterdir() if d.is_dir()])
REQUIRED_FILES = ["data.csv", "data.hdf5", "metadata.json", "README.md"]
REQUIRED_META_KEYS = ["system", "ground_truth", "initial_conditions",
                      "noise_model", "sample_rate_hz", "duration_s",
                      "n_observations", "integrator"]


@pytest.mark.parametrize("dataset", DATASETS, ids=[d.name for d in DATASETS])
def test_required_files_present(dataset):
    for fname in REQUIRED_FILES:
        assert (dataset / fname).exists(), f"Missing {fname} in {dataset.name}"


@pytest.mark.parametrize("dataset", DATASETS, ids=[d.name for d in DATASETS])
def test_metadata_schema(dataset):
    meta = json.load(open(dataset / "metadata.json"))
    for key in REQUIRED_META_KEYS:
        assert key in meta, f"Missing key '{key}' in {dataset.name}/metadata.json"
    assert meta["n_observations"] > 0
    assert meta["sample_rate_hz"] > 0
    assert meta["duration_s"] > 0


@pytest.mark.parametrize("dataset", DATASETS, ids=[d.name for d in DATASETS])
def test_csv_row_count_matches_metadata(dataset):
    meta = json.load(open(dataset / "metadata.json"))
    df   = pd.read_csv(dataset / "data.csv")
    assert len(df) == meta["n_observations"], (
        f"{dataset.name}: CSV has {len(df)} rows, metadata says {meta['n_observations']}"
    )


@pytest.mark.parametrize("dataset", DATASETS, ids=[d.name for d in DATASETS])
def test_hdf5_attributes_match_metadata(dataset):
    meta = json.load(open(dataset / "metadata.json"))
    with h5py.File(dataset / "data.hdf5", "r") as f:
        assert f.attrs["sample_rate"] == meta["sample_rate_hz"]
        assert abs(f.attrs["duration_s"] - meta["duration_s"]) < 0.01


@pytest.mark.parametrize("dataset", DATASETS, ids=[d.name for d in DATASETS])
def test_no_nan_or_inf_in_csv(dataset):
    df = pd.read_csv(dataset / "data.csv")
    numeric = df.select_dtypes(include=[np.number])
    assert not numeric.isnull().any().any(), f"{dataset.name}: NaN values in CSV"
    assert np.isfinite(numeric.values).all(),  f"{dataset.name}: Inf values in CSV"


@pytest.mark.parametrize("dataset", DATASETS, ids=[d.name for d in DATASETS])
def test_noise_std_matches_metadata(dataset):
    meta  = json.load(open(dataset / "metadata.json"))
    sigma = meta["noise_model"]["sigma"]
    df    = pd.read_csv(dataset / "data.csv")
    # Check first noisy coordinate if clean reference exists
    if "theta_rad_clean" in df.columns and "theta_rad_noisy" in df.columns:
        noise = df["theta_rad_noisy"].values - df["theta_rad_clean"].values
        assert abs(np.std(noise) - sigma) < 0.002, (
            f"{dataset.name}: noise std {np.std(noise):.4f} != expected {sigma}"
        )
