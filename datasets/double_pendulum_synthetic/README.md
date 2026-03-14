# double_pendulum_synthetic

Synthetic double pendulum dataset generated from MechanicsDSL forward simulation in the near-periodic (small-angle) regime. Designed for parameter estimation of mass and rod length from two-DOF trajectory data.

---

## Physical System

Two identical bobs of mass $m$ and rod length $l$ connected in series. Lagrangian:

$$L = \frac{1}{2}ml^2\bigl(2\dot{\theta}_1^2 + \dot{\theta}_2^2 + 2\dot{\theta}_1\dot{\theta}_2\cos(\theta_1-\theta_2)\bigr) + mgl\bigl(2\cos\theta_1 + \cos\theta_2\bigr)$$

**DSL specification:**
```
\system{double_pendulum}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\lagrangian{
    0.5*m*l^2*(2*\dot{theta1}^2 + \dot{theta2}^2
               + 2*\dot{theta1}*\dot{theta2}*cos(theta1-theta2))
    + m*g*l*(2*cos(theta1) + cos(theta2))
}
\initial{theta1: 0.3, theta2: 0.2}
\target{python_numpy}
```

---

## Ground Truth

| Parameter | Value | Unit |
|-----------|-------|------|
| m | 1.0 | kg |
| l | 1.0 | m |
| g | 9.81 | m/s² |
| θ₁₀ | 0.3 | rad |
| θ₂₀ | 0.2 | rad |
| ω₁₀, ω₂₀ | 0.0 | rad/s |

---

## Noise Model

Gaussian noise applied independently to θ₁ and θ₂ observations:
- **σ = 0.005 rad**, seed = 123

---

## Files

| File | Description |
|------|-------------|
| `data.csv` | Time series: t, θ₁_clean, θ₂_clean, ω₁_clean, ω₂_clean, θ₁_noisy, θ₂_noisy |
| `data.hdf5` | Full-precision HDF5 with ground-truth attributes |
| `metadata.json` | Machine-readable description |
| `examples/estimate_scipy.py` | Joint (m, l) estimation via least-squares |
| `examples/validate_forward.py` | Forward simulation validation |

---

## Note on Regime

Duration is limited to 5 s to stay in the near-periodic regime where parameter estimation is well-posed. Longer trajectories in the chaotic regime have degenerate likelihoods.

---

## Citation

```bibtex
@software{mechanicsdsl2026,
  author = {Parsons, Noah},
  title  = {{MechanicsDSL}},
  year   = {2026},
  doi    = {10.5281/zenodo.17771040}
}
```
