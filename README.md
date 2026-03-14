<p align="center">
  <img src="https://raw.githubusercontent.com/MechanicsDSL/mechanicsdsl/main/docs/images/logo.png" alt="MechanicsDSL Logo" width="360">
</p>

<h1 align="center">mechanicsdsl-datasets</h1>

<p align="center">
  <em>Reference datasets for physics parameter estimation and inverse problem benchmarking.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/mechanicsdsl-datasets/"><img src="https://img.shields.io/pypi/v/mechanicsdsl-datasets" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/datasets-3-blue" alt="3 Datasets">
  <img src="https://img.shields.io/badge/format-CSV%20%7C%20HDF5%20%7C%20JSON-blue" alt="Formats">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://github.com/MechanicsDSL/mechanicsdsl"><img src="https://img.shields.io/badge/core-mechanicsdsl-blue" alt="Core Package"></a>
</p>

---

## Install

```bash
pip install mechanicsdsl-datasets
```

Or clone for direct access to raw CSV/HDF5 files:

```bash
git clone https://github.com/MechanicsDSL/mechanicsdsl-datasets
```

---

## Available Datasets

| Dataset | System | Observations | Rate | Noise σ | Ground Truth |
|---------|--------|-------------|------|---------|--------------|
| [`pendulum_synthetic`](datasets/pendulum_synthetic/) | Simple pendulum | 1,001 | 100 Hz | 0.010 rad | m=1.0, l=0.5, g=9.81 |
| [`double_pendulum_synthetic`](datasets/double_pendulum_synthetic/) | Double pendulum (near-periodic) | 501 | 100 Hz | 0.005 rad | m=1.0, l=1.0, g=9.81 |
| [`coupled_oscillators_synthetic`](datasets/coupled_oscillators_synthetic/) | Coupled pendulums (3 beat periods) | 2,001 | 100 Hz | 0.005 rad | m=1.0, l=1.0, k=0.5 |

All datasets include clean trajectories, Gaussian-noisy observations, HDF5 with embedded metadata, and paired estimation/validation scripts.

---

## Quick Start (pip install)

```python
import mechanicsdsl_datasets as mds

# Load a dataset
dataset = mds.load("pendulum_synthetic")
print(dataset.ground_truth)    # {'m': 1.0, 'l': 0.5, 'g': 9.81}
print(dataset.t.shape)         # (1001,)
print(dataset.theta_noisy[:5]) # noisy observations

# Run built-in validation
dataset.validate()             # confirms forward simulation matches clean data
```

---

## Quick Start (raw files)

```python
import pandas as pd
import json

data = pd.read_csv("datasets/pendulum_synthetic/data.csv")
meta = json.load(open("datasets/pendulum_synthetic/metadata.json"))

t         = data["t_s"].values
theta_obs = data["theta_rad_noisy"].values
truth     = meta["ground_truth"]   # {'m': 1.0, 'l': 0.5, 'g': 9.81}
```

---

## Repository Structure

```
mechanicsdsl-datasets/
├── datasets/
│   ├── pendulum_synthetic/
│   │   ├── README.md, data.csv, data.hdf5, metadata.json
│   │   └── examples/  estimate_scipy.py, validate_forward.py
│   ├── double_pendulum_synthetic/
│   │   ├── README.md, data.csv, data.hdf5, metadata.json
│   │   └── examples/  estimate_scipy.py, validate_forward.py
│   └── coupled_oscillators_synthetic/
│       ├── README.md, data.csv, data.hdf5, metadata.json
│       └── examples/  estimate_normal_modes.py, validate_forward.py
├── docs/
│   ├── schema.md           # metadata.json schema reference
│   └── estimation_guide.md # least-squares, MCMC, FFT methods
├── scripts/
│   ├── generate_all.py     # regenerate all synthetic datasets
│   └── validate_all.py     # run all validate_forward.py scripts
└── tests/
    └── test_datasets.py    # schema, row count, NaN, noise std checks
```

---

## Estimation Examples

Each dataset ships with working estimation scripts:

```bash
# Estimate pendulum length from noisy observations
cd datasets/pendulum_synthetic
python examples/estimate_scipy.py
# Estimated: l = 0.4998 m  (ground truth: 0.5000 m)

# Identify normal mode frequencies via FFT
cd datasets/coupled_oscillators_synthetic
python examples/estimate_normal_modes.py
# omega1 = 3.1321 rad/s, omega2 = 3.2636 rad/s, k = 0.500 N/m
```

---

## Testing

```bash
pip install pytest h5py numpy scipy pandas
pytest tests/ -v
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Experimental datasets with real sensor data especially welcome.

## Citation

```bibtex
@dataset{mechanicsdsl_datasets2026,
  author = {Parsons, Noah},
  title  = {{MechanicsDSL Physics Datasets}},
  year   = {2026},
  doi    = {10.5281/zenodo.17771040},
  url    = {https://github.com/MechanicsDSL/mechanicsdsl-datasets}
}
```

## License

MIT License — see [LICENSE](LICENSE).
