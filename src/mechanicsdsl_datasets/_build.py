"""
_build.py
---------
Build (or rebuild) a dataset directory from its ``.msl`` spec and
``metadata.json``.

This is the writer behind every dataset added since 0.2.0 and behind
``mechanicsdsl-datasets regenerate``. Column layout (``"format": 2`` in the
metadata)::

    t_s, <q>_clean ..., <q>_dot_clean ..., <q>_noisy ...

Noise is added to the positions listed in ``noise_model.applied_to`` with
``RandomState(random_seed)`` drawn as one ``(n_channels, n_samples)`` array,
so regeneration is bit-identical given the same metadata.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from ._generate import generate_series


def build_dataset(directory: str, out_csv: Optional[str] = None) -> Dict[str, List[float]]:
    """Regenerate ``data.csv`` for a format-2 dataset directory.

    Returns the columns written. ``out_csv`` defaults to ``<directory>/data.csv``.
    """
    d = Path(directory)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("format") != 2:
        raise ValueError(f"{d.name}: only format-2 datasets can be rebuilt generically "
                         "(older datasets ship their own generate.py)")
    msl = (d / meta["dsl_specification"]).read_text(encoding="utf-8")
    tol = meta.get("tolerances", {})
    noise = meta["noise_model"]
    coords = list(meta["coordinates"])
    series = generate_series(
        msl, duration=meta["duration_s"], num_points=meta["n_observations"],
        method=meta.get("integrator"), rtol=tol.get("rtol"), atol=tol.get("atol"),
        noise_sigma=float(noise["sigma"]), noise_seed=int(meta["random_seed"]),
        noise_channels=list(noise.get("applied_to", coords)),
    )
    cols: Dict[str, List[float]] = {"t_s": list(series.t)}
    for c in coords:
        cols[f"{c}_clean"] = list(series.clean[c])
    for c in coords:
        cols[f"{c}_dot_clean"] = list(series.velocities[c])
    for c in noise.get("applied_to", coords):
        cols[f"{c}_noisy"] = list(series.noisy[c])

    path = Path(out_csv) if out_csv else d / "data.csv"
    names = list(cols)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(names)
        for i in range(len(series.t)):
            w.writerow([f"{cols['t_s'][i]:.6f}"] + [f"{cols[n][i]:.12f}" for n in names[1:]])
    return cols
