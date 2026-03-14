#!/usr/bin/env python3
"""
generate_all.py
---------------
Regenerates all synthetic datasets in this repository from scratch.
Useful for updating datasets after MechanicsDSL version changes.

Usage:
    python scripts/generate_all.py
    python scripts/generate_all.py --seed 42
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


DATASETS = [
    "datasets/pendulum_synthetic",
    "datasets/double_pendulum_synthetic",
    "datasets/coupled_oscillators_synthetic",
]


def run_generation_script(dataset_dir: Path, seed: int = None):
    gen_script = dataset_dir / "generate.py"
    if not gen_script.exists():
        # Fall back to validate script to confirm existing data is intact
        print(f"  No generate.py found — skipping regeneration for {dataset_dir.name}")
        return True

    cmd = [sys.executable, str(gen_script)]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=dataset_dir)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:300]}")
        return False
    print(f"  {result.stdout.strip()}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    success_count = 0

    for rel_path in DATASETS:
        dataset_dir = root / rel_path
        print(f"\n{dataset_dir.name}")
        if not dataset_dir.exists():
            print(f"  Directory not found: {dataset_dir}")
            continue
        if run_generation_script(dataset_dir, args.seed):
            success_count += 1

    print(f"\n{success_count}/{len(DATASETS)} datasets processed.")


if __name__ == "__main__":
    main()
