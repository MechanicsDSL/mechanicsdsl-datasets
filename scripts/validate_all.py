#!/usr/bin/env python3
"""
validate_all.py
---------------
Runs validate_forward.py for every dataset and reports results.

Usage:
    python scripts/validate_all.py
"""
import subprocess, sys
from pathlib import Path

root = Path(__file__).parent.parent
passed, failed = [], []

for validate_script in sorted(root.glob("datasets/*/examples/validate_forward.py")):
    dataset = validate_script.parent.parent.name
    print(f"Validating {dataset}...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, str(validate_script)],
        capture_output=True, text=True, cwd=validate_script.parent.parent
    )
    if result.returncode == 0 and "passed" in result.stdout.lower():
        print("✓")
        passed.append(dataset)
    else:
        print("✗")
        print(result.stdout[-300:])
        print(result.stderr[-300:])
        failed.append(dataset)

print(f"\n{len(passed)} passed, {len(failed)} failed.")
if failed:
    sys.exit(1)
