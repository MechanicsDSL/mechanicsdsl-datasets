# damped_oscillator_synthetic

A damped mass-spring oscillator: the benchmark's first **non-conservative**
system. Energy decays, so it tests whether an estimator can separate stiffness
from dissipation.

## Physical system

$$m\ddot x = -kx - b\dot x$$

Derived by `mechanicsdsl-core` from the Lagrangian $L = \tfrac12 m\dot x^2 - \tfrac12 kx^2$
plus the generalized force $-b\dot x$ ([`damped_oscillator_synthetic.msl`](damped_oscillator_synthetic.msl)).

## Ground truth

| Parameter | Value | Unit |
|-----------|-------|------|
| m | 1.0 | kg |
| k | 10.0 | N/m |
| b | 0.5 | N·s/m |
| x₀ | 1.0 | m |
| v₀ | 0.0 | m/s |

## Data

801 samples at 100 Hz over 8 s, integrated with DOP853 (rtol 1e-12, atol 1e-14).
Gaussian noise σ = 0.005 m on `x` from `RandomState(2026)`.

| Column | Meaning |
|--------|---------|
| `t_s` | time [s] |
| `x_clean` | noise-free displacement [m] |
| `x_dot_clean` | noise-free velocity [m/s] |
| `x_noisy` | observed displacement [m] |

## Identifiability

From `x(t)` alone only `k/m` and `b/m` are identifiable. Hold `m` at its true
value and estimate `k` and `b`; this is what `ds.estimable` reports and what the
baseline does.

## Reproduce

```bash
mechanicsdsl-datasets regenerate damped_oscillator_synthetic
mechanicsdsl-datasets validate damped_oscillator_synthetic
mechanicsdsl-datasets baseline damped_oscillator_synthetic
```
