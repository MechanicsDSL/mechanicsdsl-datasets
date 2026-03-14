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
    print(dataset.theta_noisy[:5])   # noisy observations

    dataset.validate()               # runs forward simulation check
    dataset.summary()                # prints dataset metadata table
"""

from mechanicsdsl_datasets._loader import load, list_datasets, Dataset
from mechanicsdsl_datasets._version import __version__

__all__ = ["load", "list_datasets", "Dataset", "__version__"]
