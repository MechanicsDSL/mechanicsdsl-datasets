"""
test_benchmark.py
-----------------
Generic validation, the new datasets, the scoring API, the reference baseline,
dataset transformations and the CLI.
"""
import json

import numpy as np
import pytest

pytest.importorskip("mechanics_dsl", reason="mechanicsdsl-core required")

import mechanicsdsl_datasets as mds  # noqa: E402
from mechanicsdsl_datasets._generate import generate_series  # noqa: E402
from mechanicsdsl_datasets.cli import main as cli_main  # noqa: E402

REPRODUCIBLE = [n for n in mds.list_datasets() if mds.dataset_info(n)["reproducible"]]


@pytest.mark.parametrize("name", REPRODUCIBLE)
def test_reproducible_datasets_validate(name):
    res = mds.load(name).validate(tol=1e-8)
    assert res.passed, res.max_error


def test_unverified_dataset_reports_honestly():
    res = mds.load("double_pendulum_synthetic").validate()
    assert not res.passed and "PROVENANCE" in res.note
    with pytest.raises(AssertionError):
        mds.load("double_pendulum_synthetic").validate(raise_on_fail=True)


def test_generate_series_indexes_core_state_by_name():
    # Regression: core interleaves [q1, q1_dot, q2, q2_dot]; positional reads
    # used to return theta1_dot as "theta2".
    ds = mds.load("coupled_oscillators_synthetic")
    s = generate_series(ds.msl, duration=1.0, num_points=11)
    assert s.clean["theta2"][0] == pytest.approx(0.0)
    assert s.clean["theta1"][0] == pytest.approx(0.3)
    assert s.velocities["theta1"][0] == pytest.approx(0.0)
    assert s.velocities["theta1"][1] < 0.0  # at t=0.1 s theta1 is falling toward 0
    assert s.clean["theta2"][1] > 0.0  # and the spring has started dragging theta2 along


def test_coupled_noise_reproduces():
    ds = mds.load("coupled_oscillators_synthetic")
    meta = ds._meta
    s = generate_series(ds.msl, duration=meta["duration_s"], num_points=meta["n_observations"],
                        method=meta["integrator"], noise_sigma=meta["noise_model"]["sigma"],
                        noise_seed=meta["random_seed"])
    for c in ("theta1", "theta2"):
        assert np.max(np.abs(s.noisy[c] - ds._data[f"{c}_rad_noisy"].values)) < 5e-10


@pytest.mark.parametrize("name", ["damped_oscillator_synthetic",
                                  "double_pendulum_chaotic_synthetic"])
def test_new_datasets_rebuild_identically(name, tmp_path):
    ds = mds.load(name)
    out = tmp_path / "rebuilt.csv"
    mds.build_dataset(str(ds._dir), out_csv=str(out))
    _assert_same_csv(out, ds._dir / "data.csv")


def _assert_same_csv(a, b, tol=1e-10):
    """Same columns and values. Exact bytes are not portable: libm and
    numpy builds differ in the last printed (12th) decimal across platforms."""
    import pandas as pd

    da, db = pd.read_csv(a), pd.read_csv(b)
    assert list(da.columns) == list(db.columns)
    assert da.shape == db.shape
    assert np.max(np.abs(da.values - db.values)) < tol


def test_damped_dataset_loses_energy():
    ds = mds.load("damped_oscillator_synthetic")
    k, m = ds.ground_truth["k"], ds.ground_truth["m"]
    E = 0.5 * m * ds.x_dot_clean ** 2 + 0.5 * k * ds.x_clean ** 2
    assert E[-1] < 0.2 * E[0]


# ------------------------------------------------------------------ scoring --
def test_score_exact_and_wrong():
    exact = mds.score("pendulum_synthetic", {"g": 9.81})
    assert exact.max_relative_error == 0.0
    off = mds.score("pendulum_synthetic", {"g": 9.81 * 1.1})
    assert off.per_parameter["g"]["relative_error"] == pytest.approx(0.1)
    assert "g" in off.table()
    with pytest.raises(KeyError):
        mds.score("pendulum_synthetic", {"gravity": 9.8})


# ----------------------------------------------------------------- baseline --
@pytest.mark.parametrize("name,tol", [
    ("pendulum_synthetic", 1e-3),
    ("coupled_oscillators_synthetic", 1e-3),
    ("damped_oscillator_synthetic", 5e-3),
    ("double_pendulum_chaotic_synthetic", 1e-3),
])
def test_baseline_recovers_ground_truth(name, tol):
    r = mds.baseline_estimate(name)
    assert r.success
    assert r.score.max_relative_error < tol, r.score.table()
    ds = mds.load(name)
    # residuals are at the noise floor when the model is right
    assert r.residual_rms < 1.2 * ds.noise_model["sigma"]
    for p, v in r.estimates.items():  # truth within 4 standard errors
        assert abs(v - ds.ground_truth[p]) < 4 * r.standard_errors[p] + 1e-12


def test_baseline_flags_model_error():
    r = mds.baseline_estimate("double_pendulum_synthetic")
    assert any("model error" in n for n in r.notes)
    assert r.residual_rms > 10 * mds.load("double_pendulum_synthetic").noise_model["sigma"]


def test_baseline_rejects_unknown_parameter():
    with pytest.raises(KeyError):
        mds.baseline_estimate("pendulum_synthetic", parameters=["nope"])


# ---------------------------------------------------------- transformations --
def test_with_noise_subsample_split(tmp_path):
    ds = mds.load("pendulum_synthetic")
    loud = ds.with_noise(0.05, seed=1)
    resid = loud.theta_rad_noisy - loud.theta_rad_clean
    assert 0.04 < resid.std() < 0.06 and loud.noise_model["sigma"] == 0.05
    assert np.array_equal(ds.theta_rad_clean, loud.theta_rad_clean)

    half = ds.subsample(2)
    assert half.n_observations == 501 and half.sample_rate_hz == 50

    train, test = ds.split(0.8)
    assert train.n_observations + test.n_observations == ds.n_observations
    assert train.t[-1] < test.t[0]

    for ext in (".csv", ".json", ".npz"):
        path = ds.export(str(tmp_path / f"out{ext}"))
        assert (tmp_path / f"out{ext}").stat().st_size > 1000, path


def test_observations_prefers_noisy():
    ds = mds.load("double_pendulum_chaotic_synthetic")
    obs = ds.observations()
    assert set(obs) == {"theta1", "theta2"}
    assert not np.array_equal(obs["theta1"], ds.theta1_clean)


# ---------------------------------------------------------------------- CLI --
def test_cli_list_info_validate(capsys):
    assert cli_main(["list"]) == 0
    out = capsys.readouterr().out
    assert "damped_oscillator_synthetic" in out and "unverified" in out
    assert cli_main(["info", "pendulum_synthetic"]) == 0
    # --all exits 0: the only failure is the known-unverified dataset
    assert cli_main(["validate", "--all"]) == 0


def test_cli_baseline_json_and_score(capsys):
    assert cli_main(["baseline", "damped_oscillator_synthetic", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["estimates"]) == {"k", "b"}
    assert cli_main(["score", "pendulum_synthetic", "g=9.79"]) == 0
    assert "2.039e-03" in capsys.readouterr().out
    assert cli_main(["score", "pendulum_synthetic", "g9.79"]) == 1


def test_cli_export_and_regenerate(tmp_path):
    out = tmp_path / "p.npz"
    assert cli_main(["export", "pendulum_synthetic", str(out), "--noise", "0.02",
                     "--subsample", "5"]) == 0
    assert len(np.load(out)["t_s"]) == 201
    regen = tmp_path / "regen.csv"
    assert cli_main(["regenerate", "damped_oscillator_synthetic", "--out", str(regen)]) == 0
    _assert_same_csv(regen, mds.load("damped_oscillator_synthetic")._dir / "data.csv")
