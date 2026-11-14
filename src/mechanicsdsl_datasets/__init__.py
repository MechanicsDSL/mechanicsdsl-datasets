"""
mechanicsdsl_datasets
---------------------
Reference physics datasets for MechanicsDSL parameter estimation
and inverse problem benchmarking.
Quick start:
    import mechanicsdsl_datasets as mds
    dataset = mds.load("pendulum_synthetic")
    print(dataset.ground_truth)      # {'m': 1.0, 'l': 0.5, 'g': 9.81}
    print(dataset.t.shape)           # (1001,)
    print(dataset.theta_rad_noisy[:5])  # noisy observations
    dataset.validate()               # re-derives the data from its .msl via core
    dataset.summary()                # prints dataset metadata table

    # Benchmark tools
    result = mds.baseline_estimate("pendulum_synthetic")   # reference estimator
    print(mds.score("pendulum_synthetic", {"g": 9.79}).table())
"""

import os
import sys

from mechanicsdsl_datasets._loader import (
    Dataset,
    ValidationResult,
    dataset_info,
    list_datasets,
    load,
)
from mechanicsdsl_datasets._benchmark import BaselineResult, Score, baseline_estimate, score
from mechanicsdsl_datasets._build import build_dataset
from mechanicsdsl_datasets._generate import (
    GeneratedSeries,
    add_noise,
    generate_series,
)
from mechanicsdsl_datasets._version import __version__

__all__ = [
    "load",
    "list_datasets",
    "Dataset",
    "ValidationResult",
    "dataset_info",
    "score",
    "Score",
    "baseline_estimate",
    "BaselineResult",
    "build_dataset",
    "generate_series",
    "add_noise",
    "GeneratedSeries",
    "__version__",
]


def _show_survey_banner():
    if (
        os.environ.get("MECHANICSDSL_NO_BANNER") or
        os.environ.get("CI") or
        os.environ.get("CONTINUOUS_INTEGRATION") or
        not sys.stdout.isatty()
    ):
        return

    import pathlib
    flag = pathlib.Path.home() / ".mechanicsdsl" / ".survey_shown"
    if flag.exists():
        return

    flag.parent.mkdir(exist_ok=True)
    flag.touch()

    print(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        " MechanicsDSL is used across 54+ countries —\n"
        " but we don't know who you are.\n"
        " 60 seconds: [https://tally.so/r/XxqOqP]\n"
        " Suppress: MECHANICSDSL_NO_BANNER=1\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

_show_survey_banner()
