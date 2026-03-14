"""
_loader.py
----------
Dataset loading API for mechanicsdsl-datasets.
Locates datasets relative to the installed package or a local clone.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import h5py
    _HDF5_AVAILABLE = True
except ImportError:
    _HDF5_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
_KNOWN_DATASETS = {
    "pendulum_synthetic": {
        "system": "simple_pendulum",
        "coords": ["theta"],
        "description": "Simple pendulum, 1001 obs at 100 Hz, σ=0.01 rad",
    },
    "double_pendulum_synthetic": {
        "system": "double_pendulum",
        "coords": ["theta1", "theta2"],
        "description": "Double pendulum (near-periodic), 501 obs at 100 Hz, σ=0.005 rad",
    },
    "coupled_oscillators_synthetic": {
        "system": "coupled_pendulums",
        "coords": ["theta1", "theta2"],
        "description": "Coupled pendulums (3 beat periods), 2001 obs at 100 Hz, σ=0.005 rad",
    },
}


def list_datasets() -> List[str]:
    """Return names of all available datasets."""
    return sorted(_KNOWN_DATASETS.keys())


def _find_dataset_dir(name: str) -> Path:
    """Locate dataset directory — package data or local clone."""
    # 1. Package data (pip install)
    pkg_dir = Path(__file__).parent / "data" / name
    if pkg_dir.exists():
        return pkg_dir

    # 2. Local clone (git clone)
    repo_root = Path(__file__).parent.parent.parent.parent
    local_dir = repo_root / "datasets" / name
    if local_dir.exists():
        return local_dir

    raise FileNotFoundError(
        f"Dataset '{name}' not found. "
        f"Available: {list_datasets()}. "
        f"If using a local clone, run from the repository root."
    )


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
        raise AttributeError(f"Dataset has no attribute '{name}'. "
                             f"Available columns: {list(self._data.columns)}")

    @property
    def columns(self) -> List[str]:
        """List all available data columns."""
        return list(self._data.columns)

    def summary(self) -> None:
        """Print a formatted summary of the dataset."""
        print(f"\nDataset: {self.name}")
        print(f"  System:         {self.system}")
        print(f"  Observations:   {self.n_observations} @ {self.sample_rate_hz} Hz")
        print(f"  Duration:       {self.duration_s} s")
        print(f"  Noise model:    {self.noise_model.get('type')} σ={self.noise_model.get('sigma')} {self.noise_model.get('unit')}")
        print(f"  Ground truth:   {self.ground_truth}")
        print(f"  Columns:        {self.columns}")

    def validate(self, tol: float = 1e-5) -> bool:
        """
        Validate by running forward simulation with ground truth parameters
        and comparing to clean trajectory.

        Returns True if max error < tol, raises AssertionError otherwise.
        """
        from scipy.integrate import solve_ivp

        gt = self.ground_truth
        ic = self.initial_conditions
        system = self.system

        if system == "simple_pendulum":
            g, l = gt["g"], gt["l"]
            def eom(t, y): return [y[1], -(g/l)*np.sin(y[0])]
            y0 = [ic.get("theta_0_rad", 0.3), ic.get("omega_0_rad_s", 0.0)]
            sol = solve_ivp(eom, (self.t[0], self.t[-1]), y0, t_eval=self.t,
                           method="DOP853", rtol=1e-10, atol=1e-12)
            clean_col = "theta_rad_clean"
            if clean_col in self.columns:
                err = np.max(np.abs(sol.y[0] - self._data[clean_col].values))
                assert err < tol, f"Validation failed: max error {err:.2e} > {tol}"
                print(f"✓ Validation passed: max error = {err:.2e} rad")
                return True

        warnings.warn(f"Validation not implemented for system '{system}'")
        return False

    def to_dict(self) -> Dict:
        """Return all data as a plain dictionary."""
        return {col: self._data[col].values for col in self.columns}


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
    >>> ds.theta_noisy.shape
    (1001,)
    """
    if name not in _KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")

    dataset_dir = _find_dataset_dir(name)

    # Load metadata
    with open(dataset_dir / "metadata.json") as f:
        meta = json.load(f)

    # Load CSV
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
