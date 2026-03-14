"""
validate_forward.py
-------------------
Validates coupled_oscillators_synthetic by integrating with ground-truth
parameters and comparing against the clean trajectory.

Usage:
    python examples/validate_forward.py
"""

import numpy as np
import pandas as pd
import json
from scipy.integrate import solve_ivp

data  = pd.read_csv("../data.csv")
meta  = json.load(open("../metadata.json"))
truth = meta["ground_truth"]
ic    = meta["initial_conditions"]
m, l, k, g = truth["m"], truth["l"], truth["k"], truth["g"]

t_obs   = data["t_s"].values
th1_ref = data["theta1_rad_clean"].values
th2_ref = data["theta2_rad_clean"].values


def eom(t, y):
    th1, th2, w1, w2 = y
    return [w1, w2,
            -(g/l)*np.sin(th1) - (k/m)*(th1-th2),
            -(g/l)*np.sin(th2) + (k/m)*(th1-th2)]


sol = solve_ivp(eom, (t_obs[0], t_obs[-1]),
                [ic["theta1_0_rad"], ic["theta2_0_rad"], ic["omega1_0"], ic["omega2_0"]],
                t_eval=t_obs, method='DOP853', rtol=1e-10, atol=1e-12)

err1 = sol.y[0] - th1_ref
err2 = sol.y[1] - th2_ref

print("Forward model vs clean trajectory")
print(f"  θ₁ max abs error : {np.max(np.abs(err1)):.2e} rad")
print(f"  θ₂ max abs error : {np.max(np.abs(err2)):.2e} rad")

noise_std = meta["noise_model"]["sigma"]
noise1 = data["theta1_rad_noisy"].values - th1_ref
noise2 = data["theta2_rad_noisy"].values - th2_ref
print(f"\nNoise std θ₁: {np.std(noise1):.5f} (expected {noise_std})")
print(f"Noise std θ₂: {np.std(noise2):.5f} (expected {noise_std})")
print("\nAll checks passed." if np.max(np.abs(err1)) < 1e-5 else "WARNING: large error.")
