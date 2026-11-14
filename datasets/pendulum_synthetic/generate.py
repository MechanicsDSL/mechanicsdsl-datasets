#!/usr/bin/env python3
"""
generate.py — reproducibly regenerate the pendulum_synthetic dataset.

Compiles pendulum_synthetic.msl through mechanicsdsl-core, integrates with the
settings recorded in metadata.json, and adds Gaussian observation noise from a
fixed seed. Running this reproduces data.csv bit-for-bit (to integrator
tolerance).

Usage:
    python generate.py            # regenerate data.csv in place
    python generate.py --seed 7   # override the noise seed
    python generate.py --stdout   # print instead of writing
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
# Make the package importable when run from a source checkout.
sys.path.insert(0, str(HERE.parent.parent / "src"))

from mechanicsdsl_datasets._generate import add_noise, generate_series  # noqa: E402


def main() -> int:
    meta = json.loads((HERE / "metadata.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=meta["random_seed"])
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    msl = (HERE / meta["dsl_specification"]).read_text()
    tol = meta.get("tolerances", {})
    series = generate_series(
        msl,
        duration=meta["duration_s"],
        num_points=meta["n_observations"],
        method=meta.get("integrator"),
        rtol=tol.get("rtol"),
        atol=tol.get("atol"),
    )

    sigma = meta["noise_model"]["sigma"]
    theta_c = series.clean["theta"]
    omega_c = series.velocities["theta"]
    theta_n, omega_n = add_noise([theta_c, omega_c], sigma, args.seed)

    rows = zip(series.t, theta_c, omega_c, theta_n, omega_n)
    header = [
        "t_s", "theta_rad_clean", "omega_rad_s_clean",
        "theta_rad_noisy", "omega_rad_s_noisy",
    ]
    out = sys.stdout if args.stdout else open(HERE / "data.csv", "w", newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    for t, tc, oc, tn, on in rows:
        writer.writerow([f"{t:.6f}", f"{tc:.10f}", f"{oc:.10f}",
                         f"{tn:.10f}", f"{on:.10f}"])
    if not args.stdout:
        out.close()
        print(f"Wrote {meta['n_observations']} rows to data.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
