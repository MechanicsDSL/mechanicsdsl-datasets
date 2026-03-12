<p align="center">
  <img src="https://raw.githubusercontent.com/MechanicsDSL/mechanicsdsl/main/docs/images/logo.png" alt="MechanicsDSL Logo" width="360">
</p>

<h1 align="center">mechanicsdsl-datasets</h1>

<p align="center">
  <em>Reference datasets for physics parameter estimation and inverse problem benchmarking.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-planned-lightgrey" alt="Status: Planned">
  <img src="https://img.shields.io/badge/format-CSV%20%7C%20HDF5%20%7C%20JSON-blue" alt="Formats">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://github.com/MechanicsDSL/mechanicsdsl"><img src="https://img.shields.io/badge/core-mechanicsdsl-blue" alt="Core Package"></a>
</p>

---

## Overview

`mechanicsdsl-datasets` provides curated reference datasets for physics parameter estimation, model validation, and inverse problem benchmarking using the MechanicsDSL framework. Each dataset includes raw observations, documented ground-truth parameters where applicable, uncertainty budgets, and paired example scripts demonstrating estimation using the MechanicsDSL JAX-backed MCMC and Sobol analysis infrastructure.

---

## Planned Dataset Categories

### Synthetic Datasets (Known Ground Truth)

Clean benchmarks for validating estimation algorithms, with exact parameters documented:

| Dataset | System | Parameters | Noise Model |
|---------|--------|-----------|-------------|
| `pendulum_clean` | Simple pendulum | m, l, g | None |
| `pendulum_noisy` | Simple pendulum | m, l, g | Gaussian, σ = 0.01 rad |
| `double_pendulum_short` | Double pendulum (non-chaotic regime) | m, l₁, l₂ | Gaussian |
| `coupled_oscillators` | Two coupled oscillators | m, k₁, k₂, k_coupling | Gaussian |
| `damped_oscillator` | Driven damped oscillator | m, c, k, F₀, ω | Gaussian |
| `orbital_kepler` | Two-body Keplerian orbit | M, semi-major axis, eccentricity | None |
| `constrained_bead` | Bead on circular wire | m, R | Gaussian |

### Experimentally Collected Datasets

Real measurements with calibrated sensor uncertainty:

| Dataset | System | Instrumentation | Notes |
|---------|--------|----------------|-------|
| `pendulum_imu` | Physical pendulum | MPU6050 IMU at 100 Hz | Includes sensor calibration file |
| `spring_mass_encoder` | Spring-mass system | Quadrature encoder + load cell | Static spring calibration included |

### High-Noise and Challenging Cases

Datasets designed to stress-test estimation methods:

| Dataset | Challenge | Notes |
|---------|-----------|-------|
| `pendulum_high_noise` | σ = 0.1 rad, sparse observations (10 Hz) | Tests robustness to observation noise |
| `double_pendulum_chaotic` | Chaotic trajectory, extreme sensitivity | Short observation windows only |
| `multimodal_posterior` | Symmetric system with degenerate parameters | MCMC posterior has multiple modes |

---

## Dataset Format

Each dataset is provided in multiple formats:

```
datasets/
└── pendulum_noisy/
    ├── README.md           # Dataset description, collection method, ground truth
    ├── data.csv            # Time, observations (human-readable)
    ├── data.hdf5           # Full precision, metadata embedded
    ├── metadata.json       # Ground truth parameters, noise model, sensor specs
    └── examples/
        ├── estimate_mcmc.py        # MCMC parameter estimation example
        ├── estimate_sobol.py       # Sobol sensitivity analysis example
        └── validate_forward.py     # Forward simulation validation
```

`metadata.json` schema:
```json
{
  "system": "simple_pendulum",
  "ground_truth": { "m": 1.0, "l": 0.5, "g": 9.81 },
  "noise_model": { "type": "gaussian", "sigma": 0.01, "unit": "rad" },
  "sample_rate_hz": 100,
  "duration_s": 10.0,
  "n_observations": 1000,
  "dsl_specification": "pendulum.msl"
}
```

---

## Usage

```bash
pip install mechanicsdsl-core

git clone https://github.com/MechanicsDSL/mechanicsdsl-datasets
cd mechanicsdsl-datasets

# Run a parameter estimation example
python datasets/pendulum_noisy/examples/estimate_mcmc.py
```

Or load directly in a notebook:

```python
import pandas as pd
import json

data = pd.read_csv("datasets/pendulum_noisy/data.csv")
metadata = json.load(open("datasets/pendulum_noisy/metadata.json"))

# Ground truth for validation
print(metadata["ground_truth"])  # {'m': 1.0, 'l': 0.5, 'g': 9.81}
```

---

## Contributing Datasets

If you have collected physical measurements you would like to contribute:

1. Provide raw time-series observations in CSV or HDF5 format
2. Document the physical system, instrumentation, and uncertainty budget
3. Include a MechanicsDSL `.msl` specification for the system
4. Open a pull request — we will add the dataset with attribution

Experimental datasets with calibrated sensors and documented collection methodology are especially welcome.

---

## Status

This repository is in the planning stage. Initial datasets will be synthetic, generated from the core package with documented ground truth, to provide clean benchmarks for the MCMC and Sobol analysis infrastructure. Experimentally collected datasets will follow. Watch this repository for updates.

---

## License

Datasets are released under the MIT License unless an individual dataset's `README.md` specifies otherwise. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://github.com/MechanicsDSL/mechanicsdsl">Core Package</a> ·
  <a href="https://mechanicsdsl.readthedocs.io">Documentation</a> ·
  <a href="https://doi.org/10.5281/zenodo.17771040">Zenodo DOI</a>
</p>
