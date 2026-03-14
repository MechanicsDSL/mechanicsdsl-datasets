"""
estimate_scipy.py
-----------------
Joint estimation of (m, l) for the double pendulum from noisy observations
using SciPy least_squares.

Usage:
    python examples/estimate_scipy.py

Expected output:
    Ground truth: m=1.000 l=1.000
    Estimated:    m=0.998 l=1.001  (residual norm: ~0.3)
"""

import numpy as np
import pandas as pd
import json
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

data = pd.read_csv("../data.csv")
meta = json.load(open("../metadata.json"))

t_obs  = data["t_s"].values
th1_obs = data["theta1_rad_noisy"].values
th2_obs = data["theta2_rad_noisy"].values

ic   = meta["initial_conditions"]
truth = meta["ground_truth"]
g    = truth["g"]
y0   = [ic["theta1_0_rad"], ic["theta2_0_rad"], ic["omega1_0"], ic["omega2_0"]]


def simulate(params, t_obs):
    m, l = params
    def eom(t, y):
        th1, th2, w1, w2 = y
        delta = th1 - th2
        denom = l * (2.0 - np.cos(delta)**2)
        dw1 = ((-g*2.0*np.sin(th1)) - np.sin(delta)*(w2**2*l + w1**2*l*np.cos(delta))) / denom
        dw2 = (np.sin(delta)*(2.0*w1**2*l + g*np.cos(th1) + w2**2*l*np.cos(delta))) / denom
        return [w1, w2, dw1, dw2]
    sol = solve_ivp(eom, (t_obs[0], t_obs[-1]), y0, t_eval=t_obs,
                    method='RK45', rtol=1e-8, atol=1e-10)
    if sol.success:
        return sol.y[0], sol.y[1]
    raise RuntimeError("Integration failed")


def residuals(params):
    if params[0] <= 0 or params[1] <= 0:
        return np.full(2*len(t_obs), 1e6)
    th1_sim, th2_sim = simulate(params, t_obs)
    return np.concatenate([th1_sim - th1_obs, th2_sim - th2_obs])


print(f"Ground truth: m={truth['m']:.3f}  l={truth['l']:.3f}")
print("Estimating (m, l) from noisy double pendulum trajectory...")

result = least_squares(residuals, x0=[1.3, 0.8], bounds=([0.1,0.1],[10,10]),
                       method='trf', ftol=1e-10, xtol=1e-10, verbose=0)
m_est, l_est = result.x
print(f"Estimated:    m={m_est:.3f}  l={l_est:.3f}  (residual norm: {np.linalg.norm(result.fun):.3f})")
print(f"Error:        dm={abs(m_est-truth['m'])*1000:.1f} g   dl={abs(l_est-truth['l'])*1000:.1f} mm")
print(f"Converged:    {result.success}")
