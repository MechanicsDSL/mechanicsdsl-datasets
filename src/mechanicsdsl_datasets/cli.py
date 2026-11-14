"""
cli.py
------
Command-line interface for ``mechanicsdsl-datasets``.

Usage
-----
    mechanicsdsl-datasets list
    mechanicsdsl-datasets info pendulum_synthetic
    mechanicsdsl-datasets validate --all
    mechanicsdsl-datasets baseline damped_oscillator_synthetic
    mechanicsdsl-datasets score pendulum_synthetic g=9.79
    mechanicsdsl-datasets export pendulum_synthetic out.npz --noise 0.02 --seed 3
    mechanicsdsl-datasets regenerate damped_oscillator_synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from ._loader import _find_dataset_dir, list_datasets, load


def _list(_: argparse.Namespace) -> int:
    from ._loader import _KNOWN_DATASETS

    print("Available datasets:")
    for name in list_datasets():
        info = _KNOWN_DATASETS[name]
        flag = "reproducible" if info["reproducible"] else "unverified"
        print(f"  {name:<36} [{flag}] {info['description']}")
    return 0


def _info(args: argparse.Namespace) -> int:
    load(args.name).summary()
    return 0


def _validate(args: argparse.Namespace) -> int:
    names = list_datasets() if args.all else args.names
    if not names:
        print("give dataset names or --all", file=sys.stderr)
        return 1
    failed_expected = 0
    for name in names:
        ds = load(name)
        res = ds.validate(tol=args.tol)
        worst = max(res.max_error.values()) if res.max_error else float("nan")
        status = "PASS" if res.passed else "FAIL"
        print(f"  {status} {name:<36} max |error| {worst:.2e}  {res.note}")
        if not res.passed and ds.reproducible:
            failed_expected += 1
    return 1 if failed_expected else 0


def _baseline(args: argparse.Namespace) -> int:
    from ._benchmark import baseline_estimate

    params = args.params.split(",") if args.params else None
    r = baseline_estimate(args.name, parameters=params)
    if args.json:
        print(json.dumps({"estimates": r.estimates, "standard_errors": r.standard_errors,
                          "fixed": r.fixed, "residual_rms": r.residual_rms,
                          "notes": r.notes}, indent=2))
        return 0 if r.success else 1
    print(f"Baseline least-squares fit on {r.dataset} ({r.nfev} model evaluations)")
    for p, v in r.estimates.items():
        print(f"  {p:<8} = {v:.6g} ± {r.standard_errors[p]:.2g}")
    print(f"  residual RMS {r.residual_rms:.4g}; fixed {r.fixed}")
    if r.score:
        print(r.score.table())
    for note in r.notes:
        print(f"  note: {note}")
    return 0 if r.success else 1


def _score(args: argparse.Namespace) -> int:
    from ._benchmark import score

    est = {}
    for item in args.estimates:
        if "=" not in item:
            print(f"bad estimate '{item}': use name=value", file=sys.stderr)
            return 1
        k, v = item.split("=", 1)
        est[k] = float(v)
    print(score(args.name, est).table())
    return 0


def _export(args: argparse.Namespace) -> int:
    ds = load(args.name)
    if args.noise is not None:
        ds = ds.with_noise(args.noise, seed=args.seed)
    if args.subsample > 1:
        ds = ds.subsample(args.subsample)
    print(f"Wrote {ds.export(args.path)} ({ds.n_observations} rows)")
    return 0


def _regenerate(args: argparse.Namespace) -> int:
    from ._build import build_dataset

    d = _find_dataset_dir(args.name)
    build_dataset(str(d), out_csv=args.out)
    print(f"Regenerated {args.out or d / 'data.csv'} from {args.name}'s .msl spec")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mechanicsdsl-datasets",
                                description="MechanicsDSL reference datasets and benchmark tools.")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("list", help="List datasets").set_defaults(func=_list)

    s = sub.add_parser("info", help="Show a dataset's metadata")
    s.add_argument("name", choices=list_datasets())
    s.set_defaults(func=_info)

    s = sub.add_parser("validate", help="Re-derive clean data from each .msl through core")
    s.add_argument("names", nargs="*")
    s.add_argument("--all", action="store_true")
    s.add_argument("--tol", type=float, default=1e-6)
    s.set_defaults(func=_validate)

    s = sub.add_parser("baseline", help="Run the reference least-squares estimator")
    s.add_argument("name", choices=list_datasets())
    s.add_argument("--params", default=None, help="Comma-separated parameters to fit")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_baseline)

    s = sub.add_parser("score", help="Score your estimates against ground truth")
    s.add_argument("name", choices=list_datasets())
    s.add_argument("estimates", nargs="+", help="name=value pairs, e.g. g=9.79")
    s.set_defaults(func=_score)

    s = sub.add_parser("export", help="Export to .csv/.json/.npz, optionally re-noised")
    s.add_argument("name", choices=list_datasets())
    s.add_argument("path")
    s.add_argument("--noise", type=float, default=None, help="Re-draw noise with this sigma")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--subsample", type=int, default=1, help="Keep every n-th sample")
    s.set_defaults(func=_export)

    s = sub.add_parser("regenerate", help="Rebuild a format-2 dataset's data.csv from its spec")
    s.add_argument("name", choices=list_datasets())
    s.add_argument("--out", default=None, help="Write here instead of overwriting data.csv")
    s.set_defaults(func=_regenerate)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
