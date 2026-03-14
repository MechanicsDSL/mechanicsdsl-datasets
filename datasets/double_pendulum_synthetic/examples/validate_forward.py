"""
validate_forward.py
-------------------
Validates double_pendulum_synthetic by integrating with ground-truth
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
m, l, g = truth["m"], truth["l"], truth["g"]

t_obs   = data["t_s"].values
th1_ref = data["theta1_rad_clean"].values
th2_ref = data["theta2_rad_clean"].values
th1_n   = data["theta1_rad_noisy"].values
th2_n   = data["theta2_rad_noisy"].values


def eom(t, y):
    th1, th2, w1, w2 = y
    delta = th1 - th2
    denom = l * (2.0 - np.cos(delta)**2)
    dw1 = ((-g*2.0*np.sin(th1)) - np.sin(delta)*(w2**2*l + w1**2*l*np.cos(delta))) / denom
    dw2 = (np.sin(delta)*(2.0*w1**2*l + g*np.cos(th1) + w2**2*l*np.cos(delta))) / denom
    return [w1, w2, dw1, dw2]


sol = solve_ivp(eom, (t_obs[0], t_obs[-1]),
                [ic["theta1_0_rad"], ic["theta2_0_rad"], ic["omega1_0"], ic["omega2_0"]],
                t_eval=t_obs, method='DOP853', rtol=1e-12, atol=1e-14)

err1 = sol.y[0] - th1_ref
err2 = sol.y[1] - th2_ref
noise1 = th1_n - th1_ref
noise2 = th2_n - th2_ref

print("Forward model vs clean trajectory")
print(f"  θ₁ max abs error : {np.max(np.abs(err1)):.2e} rad")
print(f"  θ₂ max abs error : {np.max(np.abs(err2)):.2e} rad")
print()
print("Noise characterisation")
print(f"  θ₁ noise std : {np.std(noise1):.5f} rad  (expected {meta['noise_model']['sigma']})")
print(f"  θ₂ noise std : {np.std(noise2):.5f} rad  (expected {meta['noise_model']['sigma']})")
print()
print("All checks passed." if np.max(np.abs(err1)) < 1e-6 else "WARNING: large integration error.")
