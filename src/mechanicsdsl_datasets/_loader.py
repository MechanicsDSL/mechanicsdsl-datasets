"""
_loader.py
----------
Dataset loading API for mechanicsdsl-datasets.
Locates datasets relative to the installed package or a local clone.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# ``channels`` maps each generalized coordinate to its CSV columns:
#   clean    - noise-free position          (always present)
#   velocity - noise-free velocity          (optional)
#   noisy    - observed position with noise (optional)
# ``estimable`` lists parameters that are identifiable from the observed
# channels with the others held at ground truth (e.g. for a pendulum only
# g/l is identifiable, so l is fixed and g estimated).
_KNOWN_DATASETS = {
    "pendulum_synthetic": {
        "system": "simple_pendulum",
        "coords": ["theta"],
        "description": "Simple pendulum, 1001 obs at 100 Hz, σ=0.01 rad",
        "channels": {"theta": {"clean": "theta_rad_clean", "velocity": "omega_rad_s_clean",
                               "noisy": "theta_rad_noisy"}},
        "estimable": ["g"],
        "reproducible": True,
    },
    "double_pendulum_synthetic": {
        "system": "double_pendulum",
        "coords": ["theta1", "theta2"],
        "description": "Double pendulum (near-periodic), 501 obs at 100 Hz, σ=0.005 rad",
        "channels": {
            "theta1": {"clean": "theta1_rad_clean", "velocity": "omega1_rad_s_clean",
                       "noisy": "theta1_rad_noisy"},
            "theta2": {"clean": "theta2_rad_clean", "velocity": "omega2_rad_s_clean",
                       "noisy": "theta2_rad_noisy"},
        },
        "estimable": ["g"],
        "reproducible": False,
    },
    "coupled_oscillators_synthetic": {
        "system": "coupled_pendulums",
        "coords": ["theta1", "theta2"],
        "description": "Coupled pendulums (3 beat periods), 2001 obs at 100 Hz, σ=0.005 rad",
        "channels": {
            "theta1": {"clean": "theta1_rad_clean", "noisy": "theta1_rad_noisy"},
            "theta2": {"clean": "theta2_rad_clean", "noisy": "theta2_rad_noisy"},
        },
        "estimable": ["k"],
        "reproducible": True,
    },
    "damped_oscillator_synthetic": {
        "system": "damped_oscillator",
        "coords": ["x"],
        "description": "Damped mass-spring, 801 obs at 100 Hz, σ=0.005 m (non-conservative)",
        "channels": {"x": {"clean": "x_clean", "velocity": "x_dot_clean", "noisy": "x_noisy"}},
        "estimable": ["k", "b"],
        "reproducible": True,
    },
    "double_pendulum_chaotic_synthetic": {
        "system": "double_pendulum_chaotic",
        "coords": ["theta1", "theta2"],
        "description": "Double pendulum, large angles (chaotic), 301 obs at 100 Hz, σ=0.002 rad",
        "channels": {
            "theta1": {"clean": "theta1_clean", "velocity": "theta1_dot_clean",
                       "noisy": "theta1_noisy"},
            "theta2": {"clean": "theta2_clean", "velocity": "theta2_dot_clean",
                       "noisy": "theta2_noisy"},
        },
        "estimable": ["g"],
        "reproducible": True,
    },
}


def list_datasets() -> List[str]:
    """Return names of all available datasets."""
    return sorted(_KNOWN_DATASETS.keys())


def dataset_info(name: str) -> Dict:
    """Registry entry for ``name`` (description, channels, estimable parameters)."""
    if name not in _KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    return dict(_KNOWN_DATASETS[name])


def _find_dataset_dir(name: str) -> Path:
    """Locate dataset directory: package data (pip install) or local clone."""
    pkg_dir = Path(__file__).parent / "data" / name
    if pkg_dir.exists():
        return pkg_dir

    # src/mechanicsdsl_datasets/_loader.py -> repository root is three levels up
    repo_root = Path(__file__).parent.parent.parent
    local_dir = repo_root / "datasets" / name
    if local_dir.exists():
        return local_dir

    raise FileNotFoundError(
        f"Dataset '{name}' not found. "
        f"Available: {list_datasets()}. "
        f"If using a local clone, run from the repository root."
    )


@dataclass
class ValidationResult:
    """Outcome of :meth:`Dataset.validate`."""

    dataset: str
    passed: bool
    max_error: Dict[str, float]
    tolerance: float
    note: str = ""

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# Dataset dataclass
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """Loaded MechanicsDSL physics dataset."""

    name: str
    system: str
    ground_truth: Dict[str, float]
    initial_conditions: Dict[str, float]
    noise_model: Dict
    sample_rate_hz: float
    duration_s: float
    n_observations: int
    integrator: str

    # Time vector
    t: np.ndarray = field(repr=False)

    # Observations (all available coordinates, clean and noisy)
    _data: pd.DataFrame = field(repr=False)
    _meta: Dict = field(repr=False)
    _dir: Path = field(repr=False)

    def __getattr__(self, name: str) -> np.ndarray:
        """Allow attribute access to CSV columns: dataset.theta_noisy etc."""
        if "_data" in self.__dict__ and name in self._data.columns:
            return self._data[name].values
        if "_data" in self.__dict__:
            raise AttributeError(f"Dataset has no attribute '{name}'. "
                                 f"Available columns: {list(self._data.columns)}")
        raise AttributeError(name)

    # ------------------------------------------------------------ accessors --
    @property
    def columns(self) -> List[str]:
        """List all available data columns."""
        return list(self._data.columns)

    @property
    def coordinates(self) -> List[str]:
        return list(_KNOWN_DATASETS[self.name]["coords"])

    @property
    def channels(self) -> Dict[str, Dict[str, str]]:
        """``coord -> {"clean"|"velocity"|"noisy": column}`` for this dataset."""
        return {c: dict(v) for c, v in _KNOWN_DATASETS[self.name]["channels"].items()}

    @property
    def estimable(self) -> List[str]:
        """Parameters identifiable from the observations (others held at truth)."""
        return list(_KNOWN_DATASETS[self.name]["estimable"])

    @property
    def reproducible(self) -> bool:
        """True when the clean signal re-derives from the bundled ``.msl`` spec."""
        return bool(_KNOWN_DATASETS[self.name]["reproducible"])

    @property
    def msl(self) -> Optional[str]:
        """The DSL specification the dataset was generated from, if bundled."""
        spec = self._meta.get("dsl_specification")
        if spec and (self._dir / spec).exists():
            return (self._dir / spec).read_text(encoding="utf-8")
        return None

    def observations(self, noisy: bool = True) -> Dict[str, np.ndarray]:
        """``coord -> position series`` (noisy if available and requested)."""
        out = {}
        for c, ch in self.channels.items():
            col = ch.get("noisy") if noisy and ch.get("noisy") in self._data else ch["clean"]
            out[c] = self._data[col].values
        return out

    # ------------------------------------------------------------- display ---
    def summary(self) -> None:
        """Print a formatted summary of the dataset."""
        print(f"\nDataset: {self.name}")
        print(f"  System:         {self.system}")
        print(f"  Observations:   {self.n_observations} @ {self.sample_rate_hz} Hz")
        print(f"  Duration:       {self.duration_s} s")
        print(f"  Noise model:    {self.noise_model.get('type')} "
              f"σ={self.noise_model.get('sigma')} {self.noise_model.get('unit')}")
        print(f"  Ground truth:   {self.ground_truth}")
        print(f"  Estimable:      {self.estimable}")
        print(f"  Reproducible:   {'yes' if self.reproducible else 'no (see PROVENANCE.md)'}")
        print(f"  Columns:        {self.columns}")

    # ---------------------------------------------------------- validation ---
    def validate(self, tol: float = 1e-6, raise_on_fail: bool = False) -> ValidationResult:
        """Re-derive the clean signal from the bundled ``.msl`` through core.

        The spec is compiled by ``mechanicsdsl-core`` and integrated with the
        integrator and tolerances recorded in ``metadata.json``; every clean
        column is compared with the committed data. Returns a truthy
        :class:`ValidationResult` on success. Datasets known not to reproduce
        (see ``datasets/PROVENANCE.md``) report their actual error and fail.
        """
        from ._generate import generate_series

        msl = self.msl
        if msl is None:
            res = ValidationResult(self.name, False, {}, tol, "no .msl specification bundled")
        else:
            tolr = self._meta.get("tolerances", {})
            series = generate_series(
                msl, duration=self.duration_s, num_points=self.n_observations,
                method=self._meta.get("integrator"), rtol=tolr.get("rtol"),
                atol=tolr.get("atol"),
            )
            errs = {}
            for c, ch in self.channels.items():
                errs[ch["clean"]] = float(np.max(np.abs(series.clean[c]
                                                        - self._data[ch["clean"]].values)))
                if ch.get("velocity") in self._data:
                    errs[ch["velocity"]] = float(np.max(np.abs(
                        series.velocities[c] - self._data[ch["velocity"]].values)))
            ok = max(errs.values()) < tol
            note = "" if ok or self.reproducible else (
                "known not to reproduce from its .msl; see datasets/PROVENANCE.md")
            res = ValidationResult(self.name, ok, errs, tol, note)
        if raise_on_fail and not res.passed:
            raise AssertionError(f"Validation failed for {self.name}: {res.max_error} "
                                 f"(tol {tol}) {res.note}")
        return res

    # ------------------------------------------------------- transformations --
    def _replace(self, df: pd.DataFrame, **meta_updates) -> "Dataset":
        meta = dict(self._meta)
        meta.update(meta_updates)
        df = df.reset_index(drop=True)
        return dataclasses.replace(
            self, _data=df, _meta=meta, t=df["t_s"].values, n_observations=len(df),
            noise_model=meta.get("noise_model", self.noise_model),
            sample_rate_hz=meta.get("sample_rate_hz", self.sample_rate_hz),
            duration_s=float(df["t_s"].values[-1] - df["t_s"].values[0]),
        )

    def with_noise(self, sigma: float, seed: int = 0) -> "Dataset":
        """Return a copy whose noisy columns are re-drawn: clean + N(0, sigma²)."""
        rng = np.random.RandomState(seed)
        df = self._data.copy()
        for ch in self.channels.values():
            noisy = ch.get("noisy") or ch["clean"].replace("_clean", "_noisy")
            df[noisy] = df[ch["clean"]].values + sigma * rng.normal(size=len(df))
        model = dict(self.noise_model, sigma=sigma)
        return self._replace(df, noise_model=model, random_seed=seed)

    def subsample(self, step: int) -> "Dataset":
        """Keep every ``step``-th observation (lower sample rate)."""
        if step < 1:
            raise ValueError("step must be >= 1")
        return self._replace(self._data.iloc[::step],
                             sample_rate_hz=self.sample_rate_hz / step)

    def split(self, fraction: float = 0.7) -> Tuple["Dataset", "Dataset"]:
        """Chronological train/test split (forecasting-style evaluation)."""
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be in (0, 1)")
        cut = int(round(fraction * len(self._data)))
        return self._replace(self._data.iloc[:cut]), self._replace(self._data.iloc[cut:])

    # -------------------------------------------------------------- export ---
    def to_dict(self) -> Dict:
        """Return all data as a plain dictionary."""
        return {col: self._data[col].values for col in self.columns}

    def to_dataframe(self) -> pd.DataFrame:
        return self._data.copy()

    def export(self, path: str) -> str:
        """Write the data to ``.csv``, ``.json`` or ``.npz`` (chosen by extension)."""
        p = Path(path)
        if p.suffix == ".csv":
            self._data.to_csv(p, index=False)
        elif p.suffix == ".json":
            payload = {"metadata": self._meta,
                       "data": {c: self._data[c].tolist() for c in self.columns}}
            p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        elif p.suffix == ".npz":
            np.savez_compressed(p, **self.to_dict())
        else:
            raise ValueError("export path must end in .csv, .json or .npz")
        return str(p)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load(name: str) -> Dataset:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Dataset name. Use list_datasets() to see available options.

    Returns
    -------
    Dataset
        Loaded dataset with metadata and numpy arrays.

    Examples
    --------
    >>> import mechanicsdsl_datasets as mds
    >>> ds = mds.load("pendulum_synthetic")
    >>> ds.ground_truth
    {'m': 1.0, 'l': 0.5, 'g': 9.81}
    >>> ds.theta_rad_noisy.shape
    (1001,)
    """
    if name not in _KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")

    dataset_dir = _find_dataset_dir(name)

    with open(dataset_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_csv(dataset_dir / "data.csv")
    t = df["t_s"].values

    return Dataset(
        name=name,
        system=meta["system"],
        ground_truth=meta["ground_truth"],
        initial_conditions=meta["initial_conditions"],
        noise_model=meta["noise_model"],
        sample_rate_hz=meta["sample_rate_hz"],
        duration_s=meta["duration_s"],
        n_observations=meta["n_observations"],
        integrator=meta.get("integrator", "DOP853"),
        t=t,
        _data=df,
        _meta=meta,
        _dir=dataset_dir,
    )
