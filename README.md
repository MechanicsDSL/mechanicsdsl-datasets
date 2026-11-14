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
| [`damped_oscillator_synthetic`](datasets/damped_oscillator_synthetic/) (new) | Damped mass-spring (non-conservative) | 801 | 100 Hz | 0.005 m | m=1.0, k=10.0, b=0.5 |
| [`double_pendulum_chaotic_synthetic`](datasets/double_pendulum_chaotic_synthetic/) (new) | Double pendulum, large angles (chaotic) | 301 | 100 Hz | 0.002 rad | m₁=m₂=1, l₁=l₂=1, g=9.81 |

All datasets include clean trajectories, Gaussian-noisy observations, the
`.msl` spec they come from, and a `metadata.json` with integrator, tolerances
and seed. The original three also ship HDF5 files and per-dataset
estimation/validation scripts. Four of the five reproduce from their `.msl`
through `mechanicsdsl-core`; `double_pendulum_synthetic` does not (see
[datasets/PROVENANCE.md](datasets/PROVENANCE.md)).

---

## Quick Start (pip install)

```python
import mechanicsdsl_datasets as mds

# Load a dataset
dataset = mds.load("pendulum_synthetic")
print(dataset.ground_truth)    # {'m': 1.0, 'l': 0.5, 'g': 9.81}
print(dataset.t.shape)         # (1001,)
print(dataset.theta_rad_noisy[:5])  # noisy observations

# Re-derive the clean data from the bundled .msl through mechanicsdsl-core
result = dataset.validate()    # truthy ValidationResult with per-column max error

# Variants for robustness studies
loud = dataset.with_noise(0.05, seed=1)   # re-drawn noise, same clean signal
coarse = dataset.subsample(5)             # 20 Hz instead of 100 Hz
train, test = dataset.split(0.8)          # chronological split for forecasting
dataset.export("pendulum.npz")            # also .csv / .json
```

## Benchmarking an estimator

```python
# Grade your estimates against ground truth
print(mds.score("pendulum_synthetic", {"g": 9.79}).table())

# Reference baseline: least squares through the dataset's own DSL model
r = mds.baseline_estimate("damped_oscillator_synthetic")
r.estimates, r.standard_errors      # {'k': 9.999, 'b': 0.4996}, {'k': 1.3e-3, 'b': 4.1e-4}
```

`ds.estimable` lists the parameters identifiable from the observations (for a
pendulum only g/l is, so l is held at truth and g fitted). The baseline fits in
log space, warm-starts on a short prefix of the record to avoid wrong-frequency
local minima, and reports standard errors from the Jacobian. On every
reproducible dataset it recovers the truth within four standard errors, with
residual RMS at the noise level.

The same tools from the shell:

```bash
mechanicsdsl-datasets list
mechanicsdsl-datasets validate --all
mechanicsdsl-datasets baseline pendulum_synthetic
mechanicsdsl-datasets score pendulum_synthetic g=9.79
mechanicsdsl-datasets export pendulum_synthetic noisy.csv --noise 0.02 --seed 3
mechanicsdsl-datasets regenerate damped_oscillator_synthetic
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

MIT License; see [LICENSE](LICENSE).
