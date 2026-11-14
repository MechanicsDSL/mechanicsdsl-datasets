"""
test_provenance.py
------------------
Verifies dataset provenance by re-deriving data through mechanicsdsl-core.

- pendulum_synthetic is asserted to reproduce bit-for-bit from its .msl spec
  (clean + noise), which is the strong provenance guarantee.
- All datasets are checked for physical plausibility and a noise level
  consistent with their declared sigma.

See datasets/PROVENANCE.md for the status of the two-DOF datasets.

Run with: pytest tests/test_provenance.py -v
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "mechanics_dsl", reason="mechanicsdsl-core required for provenance checks"
)

from mechanicsdsl_datasets._generate import add_noise, generate_series

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
ALL = [
    "pendulum_synthetic",
    "double_pendulum_synthetic",
    "coupled_oscillators_synthetic",
]


def _read_csv(path: Path):
    rows = [r for r in csv.reader(path.open()) if r]
    header = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    return header, data


@pytest.mark.parametrize("name", ALL)
def test_msl_spec_exists_and_compiles(name):
    d = DATASETS / name
    meta = json.loads((d / "metadata.json").read_text())
    msl = (d / meta["dsl_specification"]).read_text()
    # generate_series compiles the spec; any failure raises.
    series = generate_series(
        msl,
        duration=meta["duration_s"],
        num_points=meta["n_observations"],
        method=meta.get("integrator"),
        rtol=meta.get("tolerances", {}).get("rtol"),
        atol=meta.get("tolerances", {}).get("atol"),
    )
    assert len(series.t) == meta["n_observations"]


def test_pendulum_reproduces_bit_for_bit():
    d = DATASETS / "pendulum_synthetic"
    meta = json.loads((d / "metadata.json").read_text())
    header, data = _read_csv(d / "data.csv")

    series = generate_series(
        (d / meta["dsl_specification"]).read_text(),
        duration=meta["duration_s"],
        num_points=meta["n_observations"],
        method=meta["integrator"],
        rtol=meta["tolerances"]["rtol"],
        atol=meta["tolerances"]["atol"],
    )
    theta_c = series.clean["theta"]
    omega_c = series.velocities["theta"]
    theta_n, omega_n = add_noise(
        [theta_c, omega_c], meta["noise_model"]["sigma"], meta["random_seed"]
    )
    regen = np.column_stack([series.t, theta_c, omega_c, theta_n, omega_n])

    # Committed CSV stores 10 decimals; reproduction must match to that.
    assert np.max(np.abs(regen - data)) < 5e-10


@pytest.mark.parametrize("name", ALL)
def test_committed_data_is_physically_plausible(name):
    d = DATASETS / name
    _, data = _read_csv(d / "data.csv")
    assert np.isfinite(data).all(), "non-finite values in dataset"
    # Angles/velocities in these regimes stay O(10); guards against divergence.
    assert np.abs(data).max() < 50.0


@pytest.mark.parametrize("name", ALL)
def test_noise_level_matches_declared_sigma(name):
    d = DATASETS / name
    meta = json.loads((d / "metadata.json").read_text())
    header, data = _read_csv(d / "data.csv")
    sigma = meta["noise_model"]["sigma"]
    cols = {h: data[:, i] for i, h in enumerate(header)}
    checked = 0
    for h in header:
        if h.endswith("_noisy"):
            clean_name = h.replace("_noisy", "_clean")
            if clean_name in cols:
                resid = cols[h] - cols[clean_name]
                assert 0.5 * sigma < resid.std() < 1.5 * sigma, (
                    f"{name}/{h}: residual std {resid.std():.4f} vs sigma {sigma}"
                )
                checked += 1
    assert checked > 0, f"{name}: no matching clean/noisy column pair found"
