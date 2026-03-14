# coupled_oscillators_synthetic

Synthetic coupled pendulum dataset covering 3 beat periods. Designed for spring constant estimation and normal mode frequency identification.

---

## Physical System

Two identical pendulums of mass $m$, length $l$ coupled by a spring of stiffness $k$:

$$L = \frac{1}{2}ml^2(\dot{\theta}_1^2 + \dot{\theta}_2^2) + mgl(\cos\theta_1 + \cos\theta_2) - \frac{1}{2}kl^2(\theta_1-\theta_2)^2$$

**DSL specification:**
```
\system{coupled_pendulums}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{k}{0.5}{N/m}
\lagrangian{
    0.5*m*l^2*(\dot{theta1}^2 + \dot{theta2}^2)
    + m*g*l*(cos(theta1) + cos(theta2))
    - 0.5*k*l^2*(theta1-theta2)^2
}
\initial{theta1: 0.3, theta2: 0.0}
\target{python_numpy}
```

---

## Ground Truth

| Parameter | Value | Unit |
|-----------|-------|------|
| m | 1.0 | kg |
| l | 1.0 | m |
| k | 0.5 | N/m |
| g | 9.81 | m/s² |
| ω₁ | 3.1321 | rad/s |
| ω₂ | 3.2636 | rad/s |
| T_beat | ~48.0 | s |

---

## Noise Model

Gaussian noise applied to θ₁ and θ₂: **σ = 0.005 rad**, seed = 456

---

## Files

| File | Description |
|------|-------------|
| `data.csv` | Time series: t, θ₁_clean, θ₂_clean, θ₁_noisy, θ₂_noisy |
| `data.hdf5` | Full-precision HDF5 with all attributes including ground-truth k |
| `metadata.json` | Machine-readable description with normal mode frequencies |
| `examples/estimate_normal_modes.py` | FFT-based normal mode frequency identification |
| `examples/validate_forward.py` | Forward simulation validation |

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
