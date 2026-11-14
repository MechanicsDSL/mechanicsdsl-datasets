# Dataset provenance

Every dataset ships the DSL specification (`*.msl`) it was derived from and a
`metadata.json` recording the integrator, tolerances, noise model, and random
seed. Provenance status below reflects what has been **verified by
re-derivation through `mechanicsdsl-core`** (see `tests/test_provenance.py`;
run it yourself with `mechanicsdsl-datasets validate --all`).

| Dataset | Clean signal reproduces from `.msl`? | Noise reproduces? | Regenerator |
|---------|--------------------------------------|-------------------|-------------|
| `pendulum_synthetic` | ✅ to CSV precision (`max|Δ| ≈ 5e-11`) | ✅ `RandomState(42)` | `generate.py` |
| `coupled_oscillators_synthetic` | ✅ to CSV precision (`max|Δ| ≈ 1.5e-10`) | ✅ `RandomState(456)` | `generate_series` |
| `double_pendulum_synthetic` | ⚠️ no (diverges from the first step) | — | none |
| `damped_oscillator_synthetic` (new in 0.2.0) | ✅ to CSV precision | ✅ `RandomState(2026)` | `mechanicsdsl-datasets regenerate` |
| `double_pendulum_chaotic_synthetic` (new in 0.2.0) | ✅ to CSV precision | ✅ `RandomState(7)` | `mechanicsdsl-datasets regenerate` |

## Reproducing `pendulum_synthetic`

```bash
cd datasets/pendulum_synthetic
python generate.py            # rewrites data.csv, identical to the release
```

The clean columns are the core RK45 integration of `pendulum_synthetic.msl`
(rtol 1e-10, atol 1e-12); the noisy columns add `RandomState(42)` Gaussian
noise (σ = 0.01) drawn as a `(2, N)` array over `[theta, omega]`.

## `coupled_oscillators_synthetic`: verified in 0.2.0

Until 0.2.0 this dataset was listed as unverified. The cause was a bug in
`generate_series`, not in the data: `mechanicsdsl-core` returns the state
vector interleaved as `[theta1, theta1_dot, theta2, theta2_dot]`, but the
generator read it positionally as `[theta1, theta2, theta1_dot, theta2_dot]`,
so the regenerated "theta2" was really `theta1_dot`. With rows looked up by
name, the bundled `.msl` (DOP853, core default tolerances) reproduces both the
clean angles and the `RandomState(456)` noise to the 10 decimals the CSV stores.

## `double_pendulum_synthetic`: still unverified

With the indexing bug fixed, re-integrating the bundled standard-Lagrangian
`.msl` still does **not** reproduce the committed series: the angles already
differ by ~1e-4 rad after one 10 ms step, and by ~0.5 rad after 5 s. The
original generator therefore used different equations of motion (most likely
a relative-angle parameterisation), not merely a different integrator.

The test suite checks this dataset's physical plausibility (finite, bounded,
noise σ consistent) but not reproduction, and `ds.reproducible` is `False`.
Model-based estimators, including `baseline_estimate`, carry model error on it
(the baseline's residual RMS is ~50x the noise σ). For a double pendulum that
does reproduce, use `double_pendulum_chaotic_synthetic`.
