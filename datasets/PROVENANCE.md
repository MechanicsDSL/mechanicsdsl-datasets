# Dataset provenance

Every dataset ships the DSL specification (`*.msl`) it was derived from and a
`metadata.json` recording the integrator, tolerances, noise model, and random
seed. Provenance status below reflects what has been **verified by
re-derivation through `mechanicsdsl-core`** (see `tests/test_provenance.py`).

| Dataset | Clean signal reproduces from `.msl`? | Noise reproduces? | Regenerator |
|---------|--------------------------------------|-------------------|-------------|
| `pendulum_synthetic` | ✅ bit-identical (`max|Δ| = 0`) | ✅ `RandomState(42)` | `generate.py` |
| `double_pendulum_synthetic` | ⚠️ not reproduced by the standard Lagrangian | — | pending |
| `coupled_oscillators_synthetic` | ⚠️ not reproduced by the standard Lagrangian | — | pending |

## Reproducing `pendulum_synthetic`

```bash
cd datasets/pendulum_synthetic
python generate.py            # rewrites data.csv, bit-identical to the release
```

The clean columns are the core RK45 integration of `pendulum_synthetic.msl`
(rtol 1e-10, atol 1e-12); the noisy columns add `RandomState(42)` Gaussian
noise (σ = 0.01) drawn as a `(2, N)` array over `[theta, omega]`.

## Status of the two-DOF datasets

`double_pendulum_synthetic` and `coupled_oscillators_synthetic` include correct
standard-Lagrangian `.msl` specs, and the committed clean trajectories share the
right initial conditions and amplitude envelope. However, re-integrating those
Lagrangians through core does **not** reproduce the committed series — the two
diverge from the very first step, i.e. the original generator used a different
equation-of-motion convention (most likely a relative-angle parameterisation),
not merely a different integrator.

Until the original convention is confirmed, these two datasets are treated as
**unverified**: the test suite checks their physical plausibility (finite,
bounded, noise σ consistent) but does not assert bit-level reproduction. The
recommended fix is to regenerate them from the bundled `.msl` specs (which are
correct standard Lagrangians) once maintainers confirm it is acceptable to
supersede the currently published (DOI'd) bytes.
