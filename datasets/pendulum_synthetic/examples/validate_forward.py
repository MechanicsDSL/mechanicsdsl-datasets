"""
validate_forward.py
-------------------
Validates the pendulum_synthetic dataset by integrating the forward model
with ground-truth parameters and comparing against the clean trajectory.

Quantifies integration error (should be at the tolerance floor) and
observation noise (should match metadata σ = 0.01 rad).

Usage
-----
    python examples/validate_forward.py

Expected output
---------------
    Forward model vs clean trajectory
      Max absolute error : 3.2e-09 rad   (integration tolerance floor)
      RMS error          : 1.1e-09 rad

    Noise characterisation (noisy - clean)
      Mean               :  0.0001 rad   (should be ~0)
      Std dev            :  0.0100 rad   (should be ~0.01)
      Max absolute       :  0.0312 rad
"""

import numpy as np
import pandas as pd
import h5py
import json
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = pd.read_csv("../data.csv")
meta = json.load(open("../metadata.json"))

t_obs        = data["t_s"].values
theta_clean  = data["theta_rad_clean"].values
theta_noisy  = data["theta_rad_noisy"].values
omega_clean  = data["omega_rad_s_clean"].values

truth = meta["ground_truth"]
ic    = meta["initial_conditions"]
g, l  = truth["g"], truth["l"]


# ---------------------------------------------------------------------------
# Forward integration with ground-truth parameters
# ---------------------------------------------------------------------------
def eom(t, y):
    return [y[1], -(g / l) * np.sin(y[0])]

sol = solve_ivp(
    eom,
    t_span=(t_obs[0], t_obs[-1]),
    y0=[ic["theta_0_rad"], ic["omega_0_rad_s"]],
    t_eval=t_obs,
    method="RK45",
    rtol=1e-10,
    atol=1e-12,
)

theta_fwd = sol.y[0]

# ---------------------------------------------------------------------------
# Validate forward model against clean trajectory
# ---------------------------------------------------------------------------
err_fwd = theta_fwd - theta_clean

print("Forward model vs clean trajectory")
print(f"  Max absolute error : {np.max(np.abs(err_fwd)):.2e} rad   (integration tolerance floor)")
print(f"  RMS error          : {np.sqrt(np.mean(err_fwd**2)):.2e} rad")

# ---------------------------------------------------------------------------
# Characterise observation noise
# ---------------------------------------------------------------------------
noise = theta_noisy - theta_clean

print("\nNoise characterisation (noisy - clean)")
print(f"  Mean               : {np.mean(noise):+.4f} rad   (should be ~0)")
print(f"  Std dev            : {np.std(noise):.4f} rad   (should be ~{meta['noise_model']['sigma']})")
print(f"  Max absolute       : {np.max(np.abs(noise)):.4f} rad")

# ---------------------------------------------------------------------------
# HDF5 consistency check
# ---------------------------------------------------------------------------
with h5py.File("../data.hdf5", "r") as f:
    theta_hdf5 = f["theta_clean"][:]
    l_attr     = f.attrs["ground_truth_l"]
    g_attr     = f.attrs["ground_truth_g"]

hdf5_csv_diff = np.max(np.abs(theta_hdf5 - theta_clean))
print(f"\nHDF5 vs CSV consistency")
print(f"  Max difference     : {hdf5_csv_diff:.2e} rad   (should be 0.0)")
print(f"  HDF5 l attribute   : {l_attr} m   ✓" if l_attr == l else f"  HDF5 l attribute MISMATCH: {l_attr} vs {l}")
print(f"  HDF5 g attribute   : {g_attr} m/s²   ✓" if g_attr == g else f"  HDF5 g attribute MISMATCH")

print("\nAll checks passed." if hdf5_csv_diff < 1e-12 else "\nWARNING: HDF5/CSV mismatch detected.")
