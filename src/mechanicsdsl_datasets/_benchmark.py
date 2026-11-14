"""
_benchmark.py
-------------
Scoring and a reference baseline for the parameter-estimation benchmark.

* :func:`score` grades any estimator's output against a dataset's ground
  truth (absolute/relative error per parameter, and a summary).
* :func:`baseline_estimate` is a transparent reference estimator: nonlinear
  least squares (``scipy.optimize.least_squares``) on the noisy observations,
  with the forward model compiled from the dataset's own ``.msl`` spec by
  ``mechanicsdsl-core`` and integrated with DOP853. Parameters are fitted in
  log space (so they stay positive) and standard errors come from the
  Jacobian at the optimum.

A method beating the baseline on a dataset is a meaningful result; a method
that cannot recover the baseline's accuracy on ``pendulum_synthetic`` probably
has a bug.

Note: ``double_pendulum_synthetic`` and ``coupled_oscillators_synthetic`` were
not generated from their bundled ``.msl`` (see ``datasets/PROVENANCE.md``), so
any model-based estimator, including this baseline, carries model error on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ._loader import Dataset, load


@dataclass
class Score:
    dataset: str
    per_parameter: Dict[str, Dict[str, float]]
    mean_relative_error: float
    max_relative_error: float

    def table(self) -> str:
        lines = [f"Score for {self.dataset}",
                 f"  {'param':<10} {'estimate':>14} {'truth':>14} {'rel. error':>12}"]
        for p, row in self.per_parameter.items():
            lines.append(f"  {p:<10} {row['estimate']:>14.6g} {row['truth']:>14.6g} "
                         f"{row['relative_error']:>12.3e}")
        lines.append(f"  mean relative error {self.mean_relative_error:.3e}, "
                     f"max {self.max_relative_error:.3e}")
        return "\n".join(lines)


def score(dataset: Union[str, Dataset], estimates: Dict[str, float]) -> Score:
    """Grade ``estimates`` against ground truth.

    Only parameters present in both ``estimates`` and the ground truth are
    scored; unknown names raise ``KeyError`` so typos are not silently ignored.
    """
    ds = load(dataset) if isinstance(dataset, str) else dataset
    truth = ds.ground_truth
    unknown = [k for k in estimates if k not in truth]
    if unknown:
        raise KeyError(f"no ground truth for {unknown}; known: {sorted(truth)}")
    if not estimates:
        raise ValueError("no estimates given")
    rows = {}
    for p, est in estimates.items():
        tv = float(truth[p])
        err = float(est) - tv
        rows[p] = {"estimate": float(est), "truth": tv, "absolute_error": abs(err),
                   "relative_error": abs(err) / max(abs(tv), 1e-300)}
    rel = [r["relative_error"] for r in rows.values()]
    return Score(ds.name, rows, float(np.mean(rel)), float(np.max(rel)))


@dataclass
class BaselineResult:
    dataset: str
    estimates: Dict[str, float]
    standard_errors: Dict[str, float]
    fixed: Dict[str, float]
    residual_rms: float
    nfev: int
    success: bool
    score: Optional[Score] = None
    notes: List[str] = field(default_factory=list)


def _forward_model(ds: Dataset):
    """Compile the dataset's spec; return (system, rhs(t, y, pvec), param_order)."""
    import sympy as sp

    from ._bridge import compile_spec

    if ds.msl is None:
        raise ValueError(f"{ds.name} has no bundled .msl spec")
    system = compile_spec(ds.msl)
    order = sorted(system.parameters)
    exprs = [system.equations[f"{c}_ddot"] for c in system.coordinates]
    by_name = {}
    for e in exprs:
        by_name.update({str(s): s for s in e.free_symbols})
    state = [by_name.get(n, sp.Symbol(n)) for n in system.state_names]
    psyms = [by_name.get(p, sp.Symbol(p)) for p in order]
    fn = sp.lambdify(state + psyms, exprs, modules="numpy")
    n = system.dof

    def rhs(_t, y, pvec):
        qdd = fn(*y, *pvec)
        return np.concatenate([y[n:], np.asarray(qdd, dtype=float).reshape(n)])

    return system, rhs, order


def baseline_estimate(dataset: Union[str, Dataset], parameters: Optional[Sequence[str]] = None,
                      initial_guess: Optional[Dict[str, float]] = None,
                      perturbation: float = 1.3, rtol: float = 1e-9,
                      atol: float = 1e-11) -> BaselineResult:
    """Fit ``parameters`` by nonlinear least squares on the noisy observations.

    Parameters not being fitted are held at the ``.msl`` values. The default
    start is each fitted parameter's ``.msl`` value times ``perturbation``.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares

    ds = load(dataset) if isinstance(dataset, str) else dataset
    system, rhs, order = _forward_model(ds)
    fit = list(parameters) if parameters else ds.estimable
    missing = [p for p in fit if p not in system.parameters]
    if missing:
        raise KeyError(f"{missing} are not parameters of the {ds.name} model "
                       f"({order})")
    base = dict(system.parameters)
    guess = dict(initial_guess or {})
    obs = ds.observations(noisy=True)
    t_all = ds.t
    y0 = np.array(system.initial_state(), dtype=float)

    def make_residuals(n_keep):
        t = t_all[:n_keep]
        Y_obs = np.concatenate([obs[c][:n_keep] for c in system.coordinates])

        def residuals(theta):
            pv = dict(base)
            pv.update({p: float(np.exp(v)) for p, v in zip(fit, theta)})
            sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, args=([pv[p] for p in order],),
                            method="DOP853", rtol=rtol, atol=atol)
            if not sol.success or sol.y.shape[1] != len(t):
                return np.full_like(Y_obs, 1e3)
            return np.concatenate([sol.y[i] for i in range(system.dof)]) - Y_obs

        return residuals

    # Oscillatory fits are multimodal over long records (a wrong frequency
    # can still match the phase several periods later). Fit a short prefix
    # first, then warm-start on longer windows; try a start above and below
    # the nominal values and keep the best.
    n = len(t_all)
    windows = sorted({max(len(fit) + 5, int(n * f)) for f in (0.1, 0.3)} | {n})
    starts = [perturbation, 1.0 / perturbation] if not guess else [1.0]
    best = None
    for factor in starts:
        x = np.log([float(guess.get(p, base[p] * factor)) for p in fit])
        for w in windows:
            res = least_squares(make_residuals(min(w, n)), x, method="trf", x_scale="jac")
            x = res.x
        if best is None or res.cost < best.cost:
            best = res
    res = best
    est = {p: float(np.exp(v)) for p, v in zip(fit, res.x)}
    dof_resid = max(len(res.fun) - len(fit), 1)
    s2 = float(res.fun @ res.fun) / dof_resid
    notes = []
    try:
        cov_log = s2 * np.linalg.inv(res.jac.T @ res.jac)
        se = {p: est[p] * float(np.sqrt(max(cov_log[i, i], 0.0))) for i, p in enumerate(fit)}
    except np.linalg.LinAlgError:
        se = {p: float("nan") for p in fit}
        notes.append("Jacobian is singular: the fitted parameters are not jointly identifiable")
    if not ds.reproducible:
        notes.append("dataset does not reproduce from its .msl: expect model error")
    graded = None
    scorable = {p: v for p, v in est.items() if p in ds.ground_truth}
    if scorable:
        graded = score(ds, scorable)
    return BaselineResult(
        dataset=ds.name, estimates=est, standard_errors=se,
        fixed={p: base[p] for p in order if p not in fit},
        residual_rms=float(np.sqrt(np.mean(res.fun ** 2))), nfev=int(res.nfev),
        success=bool(res.success), score=graded, notes=notes,
    )
