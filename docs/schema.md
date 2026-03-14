# Dataset Schema

All datasets in this repository follow a consistent schema.

## Directory Structure

```
datasets/<name>/
├── README.md               # Human-readable description
├── data.csv                # Observations (human-readable)
├── data.hdf5               # Full-precision data with embedded attributes
├── metadata.json           # Machine-readable schema (see below)
└── examples/
    ├── estimate_*.py       # Parameter estimation examples
    └── validate_forward.py # Forward simulation validation
```

## metadata.json Schema

```json
{
  "system": "string",
  "description": "string",
  "ground_truth": {
    "<param>": <value>,
    ...
  },
  "initial_conditions": {
    "<coord>_0": <value>,
    ...
  },
  "noise_model": {
    "type": "gaussian | none",
    "sigma": <float>,
    "unit": "rad | m | ...",
    "applied_to": ["coord1", ...]
  },
  "sample_rate_hz": <int>,
  "duration_s": <float>,
  "n_observations": <int>,
  "integrator": "RK45 | DOP853 | ...",
  "tolerances": {
    "rtol": <float>,
    "atol": <float>
  },
  "random_seed": <int>,
  "files": {
    "data.csv": "description",
    "data.hdf5": "description"
  },
  "dsl_specification": "filename.msl",
  "generated_by": "MechanicsDSL vX.Y",
  "doi": "10.5281/zenodo.17771040"
}
```

## HDF5 Structure

```
data.hdf5
├── [attributes]
│   ├── system
│   ├── sample_rate
│   ├── duration_s
│   ├── noise_sigma
│   └── ground_truth_<param>  (one per ground-truth parameter)
└── [datasets]
    ├── t                     (time vector)
    ├── <coord>_clean         (noise-free trajectory)
    ├── <coord>_noisy         (with observation noise)
    └── ...
```

## CSV Columns

| Column | Description |
|--------|-------------|
| `t_s` | Time in seconds |
| `<coord>_rad_clean` | Clean angle in radians |
| `<coord>_rad_s_clean` | Clean angular velocity in rad/s |
| `<coord>_rad_noisy` | Noisy angle observation |
| `<coord>_rad_s_noisy` | Noisy angular velocity observation |

## Adding a New Dataset

1. Create `datasets/<name>/` with all required files
2. Follow the `metadata.json` schema above exactly
3. Add at least `validate_forward.py` to `examples/`
4. Run `pytest tests/test_datasets.py` to validate schema compliance
5. Open a pull request
