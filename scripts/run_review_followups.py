"""Master orchestrator for the code-review follow-up compute program:
  1. baseline_independent_tuning.py  (gamma/tau grid per external baseline)
  2. baseline_tuned_replication.py   (full MNIST n=10 replication at each baseline's own tuning)
  3. cifar10_tau_dp_search_extended.py (tau grid extended to 1e-4/1e-5/1e-6)
  4. run_subgaussian_cifar10.py 100 0.15 (4th gamma point for H1 confound check)
  5. c2_seed_extend.py               (C2/CBASE2_dir n=5 -> n=8)
  6. stageA_seed_extend.py           (Stage-A hp-probe n=1 -> n=3, both datasets)

Each sub-stage is independently resumable via its own already-saved-file
checks, so re-running this orchestrator after an interruption picks up where
it left off, matching this project's established pattern.
"""
import subprocess
import sys
import os

HERE = os.path.dirname(__file__)
PY = sys.executable

STAGES = [
    ["baseline_independent_tuning.py"],
    ["baseline_tuned_replication.py"],
    ["cifar10_tau_dp_search_extended.py"],
    ["run_subgaussian_cifar10.py", "100", "0.15"],
    ["c2_seed_extend.py"],
    ["stageA_seed_extend.py"],
]

for stage in STAGES:
    script = stage[0]
    args = stage[1:]
    print(f"\n{'='*70}\n=== RUNNING {script} {' '.join(args)} ===\n{'='*70}", flush=True)
    result = subprocess.run([PY, os.path.join(HERE, script)] + args)
    if result.returncode != 0:
        print(f"!!! {script} exited with code {result.returncode} -- stopping orchestrator.", flush=True)
        sys.exit(result.returncode)

print("\nAll review follow-up stages complete.", flush=True)
