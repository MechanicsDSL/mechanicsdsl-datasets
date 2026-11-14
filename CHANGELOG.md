# Changelog

## [0.2.0] - 2026-11-14

### Added
- New datasets `damped_oscillator_synthetic` (non-conservative) and `double_pendulum_chaotic_synthetic` (chaotic regime), both reproducible from their `.msl` via `build_dataset()`.
- Benchmark tools: `score()`, and `baseline_estimate()`, a least-squares reference estimator through the dataset's own DSL model with standard errors.
- `Dataset.validate()` works for every dataset: it re-derives the clean data from the `.msl` through core and returns a `ValidationResult`.
- `with_noise()`, `subsample()`, `split()`, `observations()`, `export()` (.csv/.json/.npz), and the `channels`, `estimable` and `reproducible` properties.
- `mechanicsdsl-datasets` CLI: `list`, `info`, `validate`, `baseline`, `score`, `export`, `regenerate`.

### Fixed
- `generate_series()` read core's interleaved state vector positionally, returning `q1_dot` as "q2" for every multi-DOF system. With this fixed, **`coupled_oscillators_synthetic` is verified**: clean and noisy columns reproduce to CSV precision (see `datasets/PROVENANCE.md`).
- The wheel did not include the data, so `load()` failed after `pip install`; datasets now ship as package data.
- `load()` looked for a local clone one directory above the repository.

### Changed
- `Dataset.validate()` returns a truthy `ValidationResult` instead of raising (pass `raise_on_fail=True` for the old behaviour).

### Infrastructure
- CI workflow: pytest on Python 3.9-3.12 (Linux) plus Windows and macOS, a syntax/undefined-name lint, and an sdist/wheel build check.
- Dependabot (pip and GitHub Actions), issue and PR templates, pre-commit config, `SECURITY.md`, `CODE_OF_CONDUCT.md`.
- Single-source version (`_version.py`, read by hatch); requires `mechanicsdsl-core>=2.1.4`.

## [Unreleased]

### Added
- `pendulum_synthetic` — 1,001 observations at 100 Hz, σ=0.01 rad Gaussian noise
- `double_pendulum_synthetic` — 501 observations, near-periodic regime, σ=0.005 rad
- `coupled_oscillators_synthetic` — 2,001 observations covering 3 beat periods, σ=0.005 rad
- Dataset schema documentation (`docs/schema.md`)
- Validation suite (`tests/test_datasets.py`) with 6 test categories
- `scripts/generate_all.py` and `scripts/validate_all.py`

## [0.1.0] — 2026-03-13

- Initial repository scaffold
