"""
_generate.py
------------
Reproducible dataset generation from MechanicsDSL specifications.

Each bundled dataset ships a ``.msl`` spec and a ``metadata.json`` recording
the exact integrator, tolerances, noise model and random seed. This module
turns those into data deterministically:

1. compile the ``.msl`` through ``mechanicsdsl-core``,
2. integrate with the recorded method/rtol/atol,
3. add Gaussian observation noise from ``numpy.random.RandomState(seed)`` drawn
   as ``(n_channels, n_samples)`` — the scheme the bundled datasets use.

Because generation is deterministic, :func:`generate_series` reproduces a
dataset's clean *and* noisy columns bit-for-bit given the same metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


class CoreNotInstalled(RuntimeError):
    """Raised when ``mechanicsdsl-core`` is not importable."""


@dataclass
class GeneratedSeries:
    t: np.ndarray
    clean: Dict[str, np.ndarray]        # coord -> clean signal
    noisy: Dict[str, np.ndarray]        # coord -> noisy signal (noised channels only)
    velocities: Dict[str, np.ndarray]   # coord -> clean velocity


def add_noise(
    clean_rows: Sequence[np.ndarray], sigma: float, seed: int
) -> List[np.ndarray]:
    """Add Gaussian noise to each row using the bundled datasets' RNG scheme.

    Noise is drawn as a single ``(n_channels, n_samples)`` array from
    ``RandomState(seed)`` so that channel *i* gets row *i* — reproducing the
    published data exactly.
    """
    rows = list(clean_rows)
    n_channels = len(rows)
    n_samples = len(rows[0]) if rows else 0
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, 1.0, size=(n_channels, n_samples))
    return [rows[i] + sigma * noise[i] for i in range(n_channels)]


def generate_series(
    msl: str,
    *,
    duration: float,
    num_points: int,
    method: Optional[str] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    noise_sigma: float = 0.0,
    noise_seed: int = 0,
    noise_channels: Optional[List[str]] = None,
) -> GeneratedSeries:
    """Compile ``msl`` and integrate it into a reproducible time series."""
    try:
        from mechanics_dsl import PhysicsCompiler
    except ImportError as exc:  # pragma: no cover
        raise CoreNotInstalled(
            "mechanicsdsl-core is required to generate datasets.\n"
            "Install it with: pip install mechanicsdsl-core"
        ) from exc

    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(msl)
    if not result.get("success"):
        raise ValueError(f"DSL compilation failed: {result.get('error')}")

    sim_kwargs = {}
    if method:
        sim_kwargs["method"] = method
    if rtol is not None:
        sim_kwargs["rtol"] = rtol
    if atol is not None:
        sim_kwargs["atol"] = atol

    sol = compiler.simulate(t_span=(0.0, duration), num_points=num_points, **sim_kwargs)
    if not sol.get("success"):
        raise RuntimeError(f"Simulation failed: {sol.get('message')}")

    coords = sol["coordinates"]
    y = sol["y"]
    # core's state vector is interleaved [q1, q1_dot, q2, q2_dot, ...]; look
    # rows up by name. (Indexing by position silently returned q1_dot as
    # "q2" for every multi-DOF system before 0.2.0.)
    state_vars = list(sol.get("state_vars") or
                      [v for c in coords for v in (c, f"{c}_dot")])
    clean = {c: y[state_vars.index(c)] for c in coords}
    velocities = {c: y[state_vars.index(f"{c}_dot")] for c in coords}

    channels = noise_channels or coords
    noisy: Dict[str, np.ndarray] = {}
    if noise_sigma > 0.0 and channels:
        noised = add_noise([clean[c] for c in channels], noise_sigma, noise_seed)
        noisy = {c: noised[i] for i, c in enumerate(channels)}

    return GeneratedSeries(
        t=np.asarray(sol["t"]),
        clean=clean,
        noisy=noisy,
        velocities=velocities,
    )
