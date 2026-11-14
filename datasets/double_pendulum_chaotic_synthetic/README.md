# double_pendulum_chaotic_synthetic

A double pendulum released from large angles (θ₁ = 2.5 rad, θ₂ = 2.0 rad), in
the chaotic regime. It is the hardest dataset in the set: small parameter or
initial-state errors grow exponentially, so it tests estimator robustness, not
just accuracy.

## Physical system

Point-mass double pendulum, standard absolute-angle Lagrangian
([`double_pendulum_chaotic_synthetic.msl`](double_pendulum_chaotic_synthetic.msl)):

$$L = \tfrac12(m_1+m_2)l_1^2\dot\theta_1^2 + \tfrac12 m_2 l_2^2\dot\theta_2^2
+ m_2 l_1 l_2\dot\theta_1\dot\theta_2\cos(\theta_1-\theta_2)
+ (m_1+m_2)gl_1\cos\theta_1 + m_2 g l_2\cos\theta_2$$

## Ground truth

| Parameter | Value | Unit |
|-----------|-------|------|
| m₁, m₂ | 1.0 | kg |
| l₁, l₂ | 1.0 | m |
| g | 9.81 | m/s² |

## Data

301 samples at 100 Hz over 3 s (a short horizon on purpose), integrated with
DOP853 (rtol 1e-12, atol 1e-14). Gaussian noise σ = 0.002 rad on both angles
from `RandomState(7)`.

| Column | Meaning |
|--------|---------|
| `t_s` | time [s] |
| `theta1_clean`, `theta2_clean` | noise-free angles [rad] |
| `theta1_dot_clean`, `theta2_dot_clean` | noise-free angular velocities [rad/s] |
| `theta1_noisy`, `theta2_noisy` | observed angles [rad] |

Unlike `double_pendulum_synthetic`, this dataset **reproduces exactly** from its
`.msl` spec through `mechanicsdsl-core` (see `../PROVENANCE.md`).

## Reproduce

```bash
mechanicsdsl-datasets regenerate double_pendulum_chaotic_synthetic
mechanicsdsl-datasets validate double_pendulum_chaotic_synthetic
mechanicsdsl-datasets baseline double_pendulum_chaotic_synthetic
```
