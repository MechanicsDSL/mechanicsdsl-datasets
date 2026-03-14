# Contributing Datasets

Thank you for considering a dataset contribution to mechanicsdsl-datasets.

## What Makes a Good Dataset

A good dataset contribution includes:

1. **A physical system describable in MechanicsDSL DSL notation**
2. **Reproducible data** — either synthetic (with documented seed) or experimental (with calibrated sensors)
3. **Ground truth or known parameters** — for synthetic datasets, exact values; for experimental, the best-estimate calibration
4. **Documented uncertainty** — noise model, sensor specs, or numerical tolerances
5. **Validation scripts** — at minimum a `validate_forward.py` that confirms forward simulation reproduces clean data

## Dataset Types We Welcome

| Type | Example | Notes |
|------|---------|-------|
| Synthetic | Any MechanicsDSL system | Generate with documented seed and tolerances |
| Experimental | Physical pendulum with IMU | Include sensor calibration file |
| High-noise | σ > 0.05 rad | Valuable for robustness testing |
| Multi-body | Double pendulum, coupled oscillators | Short duration to stay near-periodic |
| Constrained | Bead on wire, cart-pendulum | Include constraint violation checks |

## Submission Checklist

- [ ] `datasets/<n>/README.md` — physical system, DSL spec, ground truth, noise model
- [ ] `datasets/<n>/data.csv` — time series with labeled columns
- [ ] `datasets/<n>/data.hdf5` — full precision with embedded attributes
- [ ] `datasets/<n>/metadata.json` — follows schema in `docs/schema.md`
- [ ] `datasets/<n>/examples/validate_forward.py`
- [ ] `datasets/<n>/examples/estimate_*.py` — at least one estimation script
- [ ] `pytest tests/test_datasets.py` passes with your dataset included
- [ ] PR description explains the intended use case

## Getting Started

Fork the repository, create your dataset directory following an existing dataset as a template, and open a pull request.

Questions? Open an issue with the `dataset` label.
