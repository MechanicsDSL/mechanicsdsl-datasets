# pendulum_synthetic

Synthetic simple pendulum dataset generated from MechanicsDSL forward simulation with known ground-truth parameters and Gaussian observation noise. Intended as a clean benchmark for parameter estimation and inverse problem validation.

---

## Physical System

A simple pendulum governed by:

$$\ddot{\theta} = -\frac{g}{l}\sin\theta$$

Derived automatically from the Lagrangian:

$$L = \frac{1}{2}ml^2\dot{\theta}^2 - mgl(1 - \cos\theta)$$

**DSL specification:**
```
\system{pendulum_synthetic}
\parameter{m}{1.0}{kg}
\parameter{l}{0.5}{m}
\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*g*l*(1 - cos(theta))}
\initial{theta: 0.5, theta_dot: 0.0}
\target{python_numpy}
\solve{t_span: [0, 10], dt: 0.01}
```

---

## Ground Truth

| Parameter | Value | Unit |
|-----------|-------|------|
| m (mass) | 1.0 | kg |
| l (length) | 0.5 | m |
| g (gravity) | 9.81 | m/s² |
| θ₀ (initial angle) | 0.5 | rad |
| ω₀ (initial angular velocity) | 0.0 | rad/s |

---

## Noise Model

Gaussian noise applied independently to both θ and ω observations:

- **Type:** Gaussian, zero-mean
- **Standard deviation:** σ = 0.01 rad
- **Random seed:** 42 (reproducible)

---

## Files

| File | Description |
|------|-------------|
| `data.csv` | Human-readable time series. Columns: `t_s`, `theta_rad_clean`, `omega_rad_s_clean`, `theta_rad_noisy`, `omega_rad_s_noisy` |
| `data.hdf5` | Full-precision HDF5. Ground-truth parameters embedded as attributes. Datasets: `t`, `theta_clean`, `omega_clean`, `theta_noisy`, `omega_noisy` |
| `metadata.json` | Machine-readable dataset description: ground truth, noise model, integrator settings, file manifest |
| `examples/estimate_scipy.py` | Least-squares parameter estimation using SciPy optimize |
| `examples/validate_forward.py` | Forward simulation validation against clean trajectory |

---

## Quick Load

```python
import pandas as pd
import json

# Load observations
data = pd.read_csv("data.csv")
meta = json.load(open("metadata.json"))

t           = data["t_s"].values
theta_obs   = data["theta_rad_noisy"].values
ground_truth = meta["ground_truth"]   # {"m": 1.0, "l": 0.5, "g": 9.81}
```

```python
import h5py

with h5py.File("data.hdf5", "r") as f:
    t           = f["t"][:]
    theta_clean = f["theta_clean"][:]
    theta_noisy = f["theta_noisy"][:]
    l_true      = f.attrs["ground_truth_l"]   # 0.5
    g_true      = f.attrs["ground_truth_g"]   # 9.81
```

---

## Integration Details

| Setting | Value |
|---------|-------|
| Integrator | RK45 (SciPy solve_ivp) |
| rtol | 1×10⁻¹⁰ |
| atol | 1×10⁻¹² |
| Sample rate | 100 Hz |
| Duration | 10 s |
| Observations | 1,001 |

---

## Citation

If you use this dataset, please cite the MechanicsDSL software:

```bibtex
@software{mechanicsdsl2026,
  author  = {Parsons, Noah},
  title   = {{MechanicsDSL}: A Domain-Specific Language for Computational Physics Simulation},
  year    = {2026},
  doi     = {10.5281/zenodo.17771040},
  url     = {https://github.com/MechanicsDSL/mechanicsdsl}
}
```
