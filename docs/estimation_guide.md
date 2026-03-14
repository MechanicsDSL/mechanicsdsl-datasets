# Parameter Estimation Guide

This guide explains how to use the mechanicsdsl-datasets reference datasets for physics parameter estimation using MechanicsDSL's inverse problem infrastructure.

---

## Overview

Each dataset provides:
- **Clean trajectories** — noise-free forward simulation with ground-truth parameters
- **Noisy observations** — Gaussian noise applied to simulate real sensor data
- **Ground truth** — exact parameters for validation

The estimation task is to recover the ground-truth parameters from the noisy observations alone.

---

## Method 1: Least-Squares (SciPy)

Best for: quick estimation, well-posed systems, low noise.

```python
import pandas as pd
import json
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

data = pd.read_csv("datasets/pendulum_synthetic/data.csv")
meta = json.load(open("datasets/pendulum_synthetic/metadata.json"))

def simulate(params, t_obs):
    l = params[0]
    g = meta["ground_truth"]["g"]
    def eom(t, y): return [y[1], -(g/l)*np.sin(y[0])]
    sol = solve_ivp(eom, (t_obs[0], t_obs[-1]),
                    [meta["initial_conditions"]["theta_0_rad"], 0.0],
                    t_eval=t_obs, rtol=1e-9, atol=1e-11)
    return sol.y[0]

def residuals(params):
    return simulate(params, data["t_s"].values) - data["theta_rad_noisy"].values

result = least_squares(residuals, x0=[0.7], bounds=([0.05], [5.0]))
print(f"Estimated l = {result.x[0]:.4f} m")
```

---

## Method 2: MCMC (JAX + NumPyro)

Best for: uncertainty quantification, multi-modal posteriors, experimental data.

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Define probabilistic model
def pendulum_model(t_obs, theta_obs=None):
    l = numpyro.sample("l", dist.Uniform(0.1, 2.0))
    # ... forward simulate and compare to observations
    # See examples/ for full implementation
    pass

mcmc = MCMC(NUTS(pendulum_model), num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), t_obs=t_obs, theta_obs=theta_obs)
mcmc.print_summary()
```

---

## Method 3: FFT-Based (Frequency Domain)

Best for: coupled oscillators, systems with known frequency structure.

```python
import numpy as np

# Normal mode frequencies directly from FFT
dt = data["t_s"].values[1] - data["t_s"].values[0]
freqs = np.fft.rfftfreq(len(data), dt) * 2*np.pi
fft = np.abs(np.fft.rfft(data["theta1_rad_noisy"].values))
omega1 = freqs[np.argmax(fft[1:]) + 1]  # skip DC
# See coupled_oscillators_synthetic/examples/estimate_normal_modes.py
```

---

## Validation

Always validate estimated parameters by running the forward model and comparing to the **clean** trajectory:

```python
from scripts.validate_all import run_generation_script
# Or run directly:
# python datasets/<n>/examples/validate_forward.py
```

---

## Benchmark Results

| Dataset | Method | Parameter | Error |
|---------|--------|-----------|-------|
| pendulum_synthetic | Least-squares | l | < 1 mm |
| coupled_oscillators | FFT | k | < 1 mN/m |
| double_pendulum | Least-squares | l | < 2 mm |
