"""Master orchestrator for the Byzantine-bias-floor study:
  1. measure_heterogeneity.py  (zeta(alpha) per dataset)
  2. bias_floor_sweep.py       (main delta x alpha x seed sweep, no-DP)
  3. aggregate_bias_floor.py   (Delta(alpha,delta), confirmatory tests, scaling regression)

Each stage is independently resumable via its own already-saved-file checks, matching
this project's established orchestration pattern (scripts/run_review_followups.py).
"""
import subprocess
import sys
import os

HERE = os.path.dirname(__file__)
PY = sys.executable

STAGES = [
    ["measure_heterogeneity.py"],
    ["measure_rho_matched.py"],
    ["bias_floor_sweep.py"],
    ["aggregate_bias_floor.py"],
]

for stage in STAGES:
    script = stage[0]
    args = stage[1:]
    print(f"\n{'='*70}\n=== RUNNING {script} {' '.join(args)} ===\n{'='*70}", flush=True)
    result = subprocess.run([PY, os.path.join(HERE, script)] + args)
    if result.returncode != 0:
        print(f"!!! {script} exited with code {result.returncode} -- stopping orchestrator.", flush=True)
        sys.exit(result.returncode)

print("\nBias-floor pipeline complete.", flush=True)
