"""
estimate_scipy.py
-----------------
Least-squares parameter estimation for the simple pendulum using the
MechanicsDSL pendulum_synthetic dataset and SciPy optimize.

Estimates pendulum length l from noisy angular observations, holding
m and g fixed (m does not appear in the equation of motion; g is known).

Usage
-----
    python examples/estimate_scipy.py

Expected output (approximate)
------------------------------
    Ground truth:  l = 0.5000 m
    Initial guess: l = 0.7000 m
    Estimated:     l = 0.4998 m  (residual norm: 0.142)
"""

import numpy as np
import pandas as pd
import json
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
data = pd.read_csv("../data.csv")
meta = json.load(open("../metadata.json"))

t_obs     = data["t_s"].values
theta_obs = data["theta_rad_noisy"].values
ic        = meta["initial_conditions"]
truth     = meta["ground_truth"]

theta0 = ic["theta_0_rad"]
omega0 = ic["omega_0_rad_s"]
g      = truth["g"]


# ---------------------------------------------------------------------------
# Forward model: integrate pendulum EOM for a given l, return theta at t_obs
# ---------------------------------------------------------------------------
def simulate_pendulum(l: float, t_obs: np.ndarray) -> np.ndarray:
    """Integrate simple pendulum EOM and interpolate to observation times."""
    def eom(t, y):
        return [y[1], -(g / l) * np.sin(y[0])]

    sol = solve_ivp(
        eom,
        t_span=(t_obs[0], t_obs[-1]),
        y0=[theta0, omega0],
        t_eval=t_obs,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )
    if sol.success:
        return sol.y[0]
    raise RuntimeError(f"Integration failed: {sol.message}")


# ---------------------------------------------------------------------------
# Residual function for least_squares
# ---------------------------------------------------------------------------
def residuals(params: np.ndarray) -> np.ndarray:
    l = params[0]
    if l <= 0:
        return np.full_like(theta_obs, 1e6)
    theta_sim = simulate_pendulum(l, t_obs)
    return theta_sim - theta_obs


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
l_init   = 0.7          # deliberate offset from ground truth
l_bounds = ([0.05], [5.0])

print(f"Ground truth:  l = {truth['l']:.4f} m")
print(f"Initial guess: l = {l_init:.4f} m")
print("Running least-squares estimation...", flush=True)

result = least_squares(
    residuals,
    x0=[l_init],
    bounds=l_bounds,
    method="trf",
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-12,
    verbose=0,
)

l_est = result.x[0]
residual_norm = np.linalg.norm(result.fun)

print(f"Estimated:     l = {l_est:.4f} m  (residual norm: {residual_norm:.3f})")
print(f"Error:         Δl = {abs(l_est - truth['l'])*1000:.2f} mm")
print(f"Converged:     {result.success}  ({result.message})")
print(f"Evaluations:   {result.nfev}")
